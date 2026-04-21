// EllipsoidImporter.cs
// Place this file in Assets/Editor/ (or any subfolder named "Editor").
// Unity excludes Editor folders from builds automatically.
//
// Usage:
//   1. Import your FBX character into Unity as usual.
//   2. Open Tools → Import Ellipsoids.
//   3. Select the exported .json file.
//   4. Drag the armature root GameObject (the one containing all bones) into the field.
//   5. Click "Import Ellipsoids".
//
// Coordinate conversion:
//   Our Python exporter uses right-hand Y-up (FBX convention).
//   Unity uses left-hand Y-up.  The "Mirror X" toggle applies:
//     position  : x → -x
//     quaternion: (x, y, z, w) → (-x, y, z, w)
//   Enable this for FBX files from Blender, Maya, 3ds Max (default).
//   Disable if ellipsoids appear mirrored after import.

using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEditor;

public class EllipsoidImporter : EditorWindow
{
    // ── Inspector state ──────────────────────────────────────────────────────

    string      _jsonPath     = "";
    GameObject  _targetRoot;
    bool        _mirrorX      = true;
    bool        _clearExisting = true;
    bool        _drawAlways   = false;
    Vector2     _scroll;

    // ── Log ──────────────────────────────────────────────────────────────────

    readonly List<string> _log = new();

    // ── Window ───────────────────────────────────────────────────────────────

    [MenuItem("Tools/Import Ellipsoids")]
    public static void ShowWindow() =>
        GetWindow<EllipsoidImporter>("Ellipsoid Importer");

    void OnGUI()
    {
        GUILayout.Label("Ellipsoid Importer", EditorStyles.boldLabel);
        GUILayout.Space(6);

        // ── JSON file ──
        EditorGUILayout.BeginHorizontal();
        _jsonPath = EditorGUILayout.TextField("JSON File", _jsonPath);
        if (GUILayout.Button("…", GUILayout.Width(28)))
            _jsonPath = EditorUtility.OpenFilePanel("Select Ellipsoid JSON", "", "json");
        EditorGUILayout.EndHorizontal();

        // ── Armature root ──
        _targetRoot = (GameObject)EditorGUILayout.ObjectField(
            "Armature Root", _targetRoot, typeof(GameObject), allowSceneObjects: true);

        GUILayout.Space(8);
        GUILayout.Label("Options", EditorStyles.boldLabel);

        _mirrorX = EditorGUILayout.Toggle("Mirror X  (FBX → Unity)", _mirrorX);
        EditorGUI.indentLevel++;
        EditorGUILayout.HelpBox(
            "Converts right-hand Y-up (FBX/Blender/Maya) to Unity's left-hand Y-up.\n" +
            "Disable if ellipsoids appear mirrored.", MessageType.None);
        EditorGUI.indentLevel--;

        _clearExisting = EditorGUILayout.Toggle("Clear Existing Ellipsoids", _clearExisting);
        _drawAlways    = EditorGUILayout.Toggle("Draw Gizmos Always",        _drawAlways);

        GUILayout.Space(10);

        bool ready = !string.IsNullOrEmpty(_jsonPath) && _targetRoot != null;
        using (new EditorGUI.DisabledScope(!ready))
        {
            if (GUILayout.Button("Import Ellipsoids", GUILayout.Height(32)))
                RunImport();
        }

        if (!ready)
            EditorGUILayout.HelpBox(
                "Select a JSON file and the armature root GameObject.", MessageType.Warning);

        // ── Log ──
        if (_log.Count > 0)
        {
            GUILayout.Space(8);
            GUILayout.Label("Log", EditorStyles.boldLabel);
            _scroll = EditorGUILayout.BeginScrollView(_scroll, GUILayout.Height(120));
            foreach (string line in _log)
                GUILayout.Label(line, EditorStyles.miniLabel);
            EditorGUILayout.EndScrollView();
        }
    }

    // ── Import logic ─────────────────────────────────────────────────────────

    void RunImport()
    {
        _log.Clear();

        // ── Parse JSON ──
        EllipsoidFile file;
        try
        {
            string json = File.ReadAllText(_jsonPath);
            file = JsonUtility.FromJson<EllipsoidFile>(json);
        }
        catch (Exception ex)
        {
            EditorUtility.DisplayDialog("Error", $"Failed to read JSON:\n{ex.Message}", "OK");
            return;
        }

        if (file?.ellipsoids == null || file.ellipsoids.Count == 0)
        {
            EditorUtility.DisplayDialog("Error", "No ellipsoids found in JSON.", "OK");
            return;
        }

        Log($"JSON v{file.version}: {file.count} ellipsoids  |  coord={file.coordinate_system}");

        // ── Build bone map ──
        var boneMap = BuildBoneMap(_targetRoot.transform);
        Log($"Bone map: {boneMap.Count} transforms found under '{_targetRoot.name}'");

        // ── Clear existing ──
        if (_clearExisting)
        {
            var existing = _targetRoot.GetComponentsInChildren<EllipsoidData>(includeInactive: true);
            foreach (var e in existing)
                Undo.DestroyObjectImmediate(e.gameObject);
            if (existing.Length > 0)
                Log($"Removed {existing.Length} existing ellipsoids.");
        }

        // ── Create ellipsoids ──
        int imported = 0;
        int missing  = 0;

        foreach (var entry in file.ellipsoids)
        {
            if (!boneMap.TryGetValue(entry.bone, out Transform bone))
            {
                Log($"  WARNING  bone not found: '{entry.bone}'");
                missing++;
                continue;
            }

            Vector3    localPos = ConvertPosition(entry.local_center);
            Quaternion localRot = ConvertRotation(entry.local_rotation);
            Vector3    radii    = new Vector3(entry.radii[0], entry.radii[1], entry.radii[2]);

            var go = new GameObject($"Ellipsoid_{imported:D3}_{entry.bone}");
            Undo.RegisterCreatedObjectUndo(go, "Import Ellipsoid");

            go.transform.SetParent(bone, worldPositionStays: false);
            go.transform.localPosition = localPos;
            go.transform.localRotation = localRot;
            go.transform.localScale    = Vector3.one;

            var comp         = go.AddComponent<EllipsoidData>();
            comp.radii       = radii;
            comp.boneName    = entry.bone;
            comp.drawAlways  = _drawAlways;

            imported++;
        }

        Log($"Done — {imported} imported, {missing} bones not found.");
        EditorUtility.DisplayDialog(
            "Import Complete",
            $"Imported {imported} ellipsoids.\n" +
            (missing > 0 ? $"{missing} bones not found — check the log." : "All bones matched."),
            "OK");
    }

    // ── Coordinate conversion ─────────────────────────────────────────────────

    Vector3 ConvertPosition(float[] p)
    {
        // right-hand Y-up → Unity left-hand Y-up: negate X
        return _mirrorX
            ? new Vector3(-p[0], p[1], p[2])
            : new Vector3( p[0], p[1], p[2]);
    }

    Quaternion ConvertRotation(float[] q)
    {
        // q = [x, y, z, w]  (Python xyzw convention)
        // Mirror-X flips the X component of the rotation axis → negate qx
        return _mirrorX
            ? new Quaternion(-q[0], q[1], q[2], q[3])
            : new Quaternion( q[0], q[1], q[2], q[3]);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    static Dictionary<string, Transform> BuildBoneMap(Transform root)
    {
        var map = new Dictionary<string, Transform>(StringComparer.Ordinal);
        foreach (Transform t in root.GetComponentsInChildren<Transform>(includeInactive: true))
        {
            // First occurrence wins (matches FBX import behaviour for duplicate names)
            if (!map.ContainsKey(t.name))
                map[t.name] = t;
        }
        return map;
    }

    void Log(string msg)
    {
        _log.Add(msg);
        Debug.Log($"[EllipsoidImporter] {msg}");
    }

    // ── Serialisable JSON types ───────────────────────────────────────────────

    [Serializable]
    class EllipsoidFile
    {
        public int    version;
        public string coordinate_system;
        public int    count;
        public List<EllipsoidEntry> ellipsoids;
    }

    [Serializable]
    class EllipsoidEntry
    {
        public string   bone;
        public float[]  local_center;    // [x, y, z]
        public float[]  radii;           // [rx, ry, rz]
        public float[]  local_rotation;  // [x, y, z, w]
    }
}
