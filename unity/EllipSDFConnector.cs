// EllipSDFConnector.cs
// Place under Assets/Scripts/ (NOT in an Editor folder) — this is a runtime component.
//
// Drop this on a GameObject that has a MeshFilter or a SkinnedMeshRenderer
// (it auto-detects which). At runtime, call Fit() (or use the inspector
// context-menu "Fit Ellipsoids") to:
//   1. Make sure an EllipSDF instance is reachable (launch one if not).
//   2. POST the mesh (+ rig, if skinned) to the HTTP API.
//   3. Watch the fit (Debug mode logs progress; Performance mode waits silently).
//   4. Spawn one "Sphere_<i>" GameObject per fitted ellipsoid, each carrying an
//      EllipsoidData component, parented under its bone (rigged) or under this
//      object (flat).
//
// Coordinate handling: the mesh is sent to EllipSDF *unchanged* in Unity space.
// EllipSDF fits and returns ellipsoids in that same space, so results are placed
// directly with no handedness mirror. Triangle winding is flipped by default so
// the SDF inside/outside sign is correct for Unity's left-handed coords.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using Debug = UnityEngine.Debug;

[DisallowMultipleComponent]
public class EllipSDFConnector : MonoBehaviour
{
    public enum FitMode
    {
        Performance, // poll quietly, only report the final result
        Debug,       // log step/loss/count as the fit runs
    }

    [Header("Server")]
    [Tooltip("Host the EllipSDF API listens on.")]
    public string host = "127.0.0.1";
    [Tooltip("Port the EllipSDF API listens on (matches --port).")]
    public int port = 8765;
    [Tooltip("If no server answers /ping, launch one with the settings below.")]
    public bool autoLaunch = true;

    [Header("Auto-launch (only if autoLaunch)")]
    [Tooltip("Python executable. Use the venv python, e.g. " +
             "<EllipSDF>/.venv/Scripts/python.exe, or just 'python' if on PATH.")]
    public string pythonExe = "python";
    [Tooltip("Absolute path to the EllipSDF project folder (the one with main.py).")]
    public string ellipSdfDir = "";
    [Tooltip("Seconds to wait for a freshly launched server to answer /ping.")]
    public float launchTimeoutSec = 30f;

    [Header("Fit options")]
    public int numEllipsoids = 20;
    public int maxEllipsoids = 60;
    public int numSteps = 2000;
    public bool symmetry = false;
    public FitMode mode = FitMode.Debug;
    [Tooltip("Reverse triangle winding before sending. Normally leave OFF: " +
             "EllipSDF re-orients faces outward on its end, so the SDF sign is " +
             "correct either way. Only a workaround for unusual meshes.")]
    public bool flipWinding = false;

    [Header("Mesh source")]
    [Tooltip("Optional: drag the GameObject that holds the mesh " +
             "(SkinnedMeshRenderer or MeshFilter), or one of its parents. " +
             "Leave empty to auto-search this object's own hierarchy.")]
    public GameObject meshSource;

    [Header("Output")]
    [Tooltip("Delete previously spawned Sphere_* children before placing new ones.")]
    public bool clearPreviousResult = true;

    bool _busy;
    Coroutine _fitCoroutine;
    UnityWebRequest _activeReq;

    // ── Public entry points ─────────────────────────────────────────────────

    [ContextMenu("Fit Ellipsoids")]
    public void Fit()
    {
        // Re-calling Fit cancels any in-flight run and starts a fresh one.
        if (_busy) Cancel();
        _fitCoroutine = StartCoroutine(FitRoutine());
    }

    [ContextMenu("Cancel Fit")]
    public void Cancel()
    {
        if (!_busy && _fitCoroutine == null) return;

        // StopCoroutine skips the routine's finally/using, so reset state and
        // abort the in-flight web request by hand.
        if (_fitCoroutine != null) StopCoroutine(_fitCoroutine);
        _fitCoroutine = null;
        if (_activeReq != null)
        {
            _activeReq.Abort();
            _activeReq.Dispose();
            _activeReq = null;
        }
        _busy = false;
        Debug.Log("[EllipSDF] Previous fit canceled.");
    }

    public IEnumerator FitRoutine()
    {
        _busy = true;
        try
        {
            // 1. Make sure a server is reachable.
            bool up = false;
            yield return Ping(ok => up = ok);
            if (!up)
            {
                if (!autoLaunch)
                {
                    Debug.LogError($"[EllipSDF] No server at {BaseUrl} and " +
                                   "autoLaunch is off.");
                    yield break;
                }
                if (!TryLaunchServer()) yield break;
                yield return WaitForServer(launchTimeoutSec, ok => up = ok);
                if (!up)
                {
                    Debug.LogError("[EllipSDF] Server did not come up in time.");
                    yield break;
                }
            }

            // 2. Build the request body from the mesh on this GameObject.
            bool rigged;
            string body = BuildRequestJson(out rigged);
            if (body == null) yield break;

            // 3. POST /fit.
            string jobId = null;
            yield return PostFit(body, id => jobId = id);
            if (jobId == null) yield break;
            Debug.Log($"[EllipSDF] Job {jobId} started — watch the EllipSDF window.");

            // 4. Poll until done.
            FitResult result = null;
            yield return PollUntilDone(jobId, r => result = r);
            if (result == null) yield break;

            // 5. Place the ellipsoids.
            PlaceResult(result);
            Debug.Log($"[EllipSDF] Done — placed {result.count} ellipsoid(s) " +
                      $"({(result.rigged ? "rigged" : "flat")}).");
        }
        finally
        {
            _busy = false;
            _fitCoroutine = null;
            _activeReq = null;
        }
    }

    string BaseUrl => $"http://{host}:{port}";

    // Wraps SendWebRequest so a cancel can abort the in-flight request.
    IEnumerator Send(UnityWebRequest req)
    {
        _activeReq = req;
        yield return req.SendWebRequest();
        _activeReq = null;
    }

    // ── Server discovery / launch ───────────────────────────────────────────

    IEnumerator Ping(Action<bool> done)
    {
        using (var req = UnityWebRequest.Get($"{BaseUrl}/ping"))
        {
            req.timeout = 2;
            yield return Send(req);
            done(req.result == UnityWebRequest.Result.Success);
        }
    }

    bool TryLaunchServer()
    {
        if (string.IsNullOrEmpty(ellipSdfDir) ||
            !File.Exists(Path.Combine(ellipSdfDir, "main.py")))
        {
            Debug.LogError("[EllipSDF] ellipSdfDir does not point at a folder " +
                           "containing main.py — cannot auto-launch.");
            return false;
        }
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"\"{Path.Combine(ellipSdfDir, "main.py")}\" " +
                            $"--server --port {port}",
                WorkingDirectory = ellipSdfDir,
                UseShellExecute = true, // visible window the user can watch
            };
            Process.Start(psi);
            Debug.Log("[EllipSDF] Launched a new EllipSDF window.");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"[EllipSDF] Failed to launch EllipSDF: {e.Message}");
            return false;
        }
    }

    IEnumerator WaitForServer(float timeoutSec, Action<bool> done)
    {
        float deadline = Time.realtimeSinceStartup + timeoutSec;
        while (Time.realtimeSinceStartup < deadline)
        {
            bool up = false;
            yield return Ping(ok => up = ok);
            if (up) { done(true); yield break; }
            yield return new WaitForSeconds(0.5f);
        }
        done(false);
    }

    // ── Request building ────────────────────────────────────────────────────

    // Resolve a component: explicit meshSource first, then this object, its
    // children, and finally the whole character hierarchy (transform.root).
    // Both BuildRequestJson and PlaceResult use this so they always agree on
    // which renderer is the target.
    T FindInHierarchy<T>() where T : Component
    {
        if (meshSource != null)
        {
            var hit = meshSource.GetComponent<T>()
                ?? meshSource.GetComponentInChildren<T>(includeInactive: true);
            if (hit != null) return hit;
        }
        return GetComponent<T>()
            ?? GetComponentInChildren<T>(includeInactive: true)
            ?? transform.root.GetComponentInChildren<T>(includeInactive: true);
    }

    string BuildRequestJson(out bool rigged)
    {
        rigged = false;

        var smr = FindInHierarchy<SkinnedMeshRenderer>();
        var mf = FindInHierarchy<MeshFilter>();

        Mesh mesh = null;
        if (smr != null && smr.sharedMesh != null) mesh = smr.sharedMesh;
        else if (mf != null && mf.sharedMesh != null) mesh = mf.sharedMesh;

        if (mesh == null)
        {
            string smrInfo = smr == null ? "none found in hierarchy"
                : smr.sharedMesh == null ? $"found on '{smr.name}' but sharedMesh is null"
                : "found";
            string mfInfo = mf == null ? "none found in hierarchy"
                : mf.sharedMesh == null ? $"found on '{mf.name}' but sharedMesh is null"
                : "found";
            Debug.LogError(
                $"[EllipSDF] No usable mesh found in the '{transform.root.name}' hierarchy " +
                $"(searched from EllipSDFConnector on '{name}'). " +
                $"SkinnedMeshRenderer: {smrInfo}. MeshFilter: {mfInfo}.", this);
            return null;
        }

        Vector3[] verts = mesh.vertices;
        int[] tris = mesh.triangles;

        var sb = new StringBuilder(verts.Length * 24);
        sb.Append('{');

        // vertices
        sb.Append("\"vertices\":[");
        for (int i = 0; i < verts.Length; i++)
        {
            if (i > 0) sb.Append(',');
            Vector3 v = verts[i];
            AppendVec3(sb, v.x, v.y, v.z);
        }
        sb.Append(']');

        // faces (optionally winding-flipped)
        sb.Append(",\"faces\":[");
        for (int t = 0; t < tris.Length; t += 3)
        {
            if (t > 0) sb.Append(',');
            int a = tris[t], b = tris[t + 1], c = tris[t + 2];
            if (flipWinding) { int tmp = b; b = c; c = tmp; }
            sb.Append('[').Append(a).Append(',').Append(b).Append(',')
              .Append(c).Append(']');
        }
        sb.Append(']');

        // options
        sb.Append(",\"options\":{")
          .Append("\"num_ellipsoids\":").Append(numEllipsoids)
          .Append(",\"max_ellipsoids\":").Append(maxEllipsoids)
          .Append(",\"num_steps\":").Append(numSteps)
          .Append(",\"symmetry\":").Append(symmetry ? "true" : "false")
          .Append('}');

        // rig (skinned only)
        if (smr != null && smr.sharedMesh != null && smr.bones != null &&
            smr.bones.Length > 0)
        {
            AppendRig(sb, smr, mesh);
            rigged = true;
        }

        sb.Append('}');
        return sb.ToString();
    }

    void AppendRig(StringBuilder sb, SkinnedMeshRenderer smr, Mesh mesh)
    {
        Transform[] bones = smr.bones;
        Matrix4x4[] bind = mesh.bindposes;
        BoneWeight[] bw = mesh.boneWeights;

        // Transform → index, so each bone's parent index (nearest ancestor that
        // is itself a bone) can be emitted for skeleton visualization.
        var boneIndex = new Dictionary<Transform, int>();
        for (int i = 0; i < bones.Length; i++)
            if (bones[i] != null) boneIndex[bones[i]] = i;

        // bones: bind transform expressed in mesh-local space (= bindpose⁻¹),
        // matching the vertex space we sent above.
        sb.Append(",\"rig\":{\"bones\":[");
        for (int i = 0; i < bones.Length; i++)
        {
            if (i > 0) sb.Append(',');
            Matrix4x4 m = (i < bind.Length) ? bind[i].inverse : Matrix4x4.identity;
            Vector3 p = m.GetColumn(3);
            Quaternion q = m.rotation;
            int parent = -1;
            for (Transform t = bones[i] != null ? bones[i].parent : null;
                 t != null; t = t.parent)
                if (boneIndex.TryGetValue(t, out int pi)) { parent = pi; break; }
            sb.Append("{\"name\":\"").Append(Escape(bones[i] != null
                ? bones[i].name : $"Bone_{i}")).Append("\",\"position\":");
            AppendVec3(sb, p.x, p.y, p.z);
            sb.Append(",\"rotation\":");
            AppendVec4(sb, q.x, q.y, q.z, q.w);
            sb.Append(",\"parent\":").Append(parent);
            sb.Append('}');
        }
        sb.Append(']');

        // boneIndices / boneWeights — K=4 per vertex.
        sb.Append(",\"boneIndices\":[");
        for (int i = 0; i < bw.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append('[').Append(bw[i].boneIndex0).Append(',')
              .Append(bw[i].boneIndex1).Append(',')
              .Append(bw[i].boneIndex2).Append(',')
              .Append(bw[i].boneIndex3).Append(']');
        }
        sb.Append("],\"boneWeights\":[");
        for (int i = 0; i < bw.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append('[');
            AppendFloat(sb, bw[i].weight0); sb.Append(',');
            AppendFloat(sb, bw[i].weight1); sb.Append(',');
            AppendFloat(sb, bw[i].weight2); sb.Append(',');
            AppendFloat(sb, bw[i].weight3);
            sb.Append(']');
        }
        sb.Append("]}");
    }

    // ── HTTP: post / poll / fetch ───────────────────────────────────────────

    IEnumerator PostFit(string body, Action<string> done)
    {
        using (var req = new UnityWebRequest($"{BaseUrl}/fit", "POST"))
        {
            req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            yield return Send(req);

            if (req.result != UnityWebRequest.Result.Success ||
                req.responseCode != 202)
            {
                Debug.LogError($"[EllipSDF] /fit rejected: {req.responseCode} " +
                               $"{req.downloadHandler.text}");
                done(null);
                yield break;
            }
            var resp = JsonUtility.FromJson<FitResp>(req.downloadHandler.text);
            done(resp != null ? resp.job_id : null);
        }
    }

    IEnumerator PollUntilDone(string jobId, Action<FitResult> done)
    {
        int lastStep = -1;
        while (true)
        {
            yield return new WaitForSeconds(0.5f);

            StatusResp st = null;
            using (var req = UnityWebRequest.Get($"{BaseUrl}/fit/{jobId}/status"))
            {
                yield return Send(req);
                if (req.result == UnityWebRequest.Result.Success)
                    st = JsonUtility.FromJson<StatusResp>(req.downloadHandler.text);
            }
            if (st == null) continue;

            if (mode == FitMode.Debug && st.state == "running" &&
                st.step != lastStep)
            {
                lastStep = st.step;
                Debug.Log($"[EllipSDF] step {st.step}/{st.total} " +
                          $"loss={st.loss:F6} count={st.count}");
            }

            if (st.state == "error")
            {
                Debug.LogError($"[EllipSDF] Fit failed: {st.error}");
                done(null);
                yield break;
            }
            if (st.state == "done") break;
        }

        // Fetch the result.
        using (var req = UnityWebRequest.Get($"{BaseUrl}/fit/{jobId}/result"))
        {
            yield return Send(req);
            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[EllipSDF] /result failed: {req.responseCode} " +
                               $"{req.downloadHandler.text}");
                done(null);
                yield break;
            }
            done(JsonUtility.FromJson<FitResult>(req.downloadHandler.text));
        }
    }

    // ── Result placement ────────────────────────────────────────────────────

    void PlaceResult(FitResult result)
    {
        if (clearPreviousResult) ClearPrevious();
        if (result == null || result.ellipsoids == null) return;

        Dictionary<string, Transform> boneByName = null;
        if (result.rigged)
        {
            boneByName = new Dictionary<string, Transform>();
            var smr = FindInHierarchy<SkinnedMeshRenderer>();
            if (smr != null && smr.bones != null)
                foreach (var b in smr.bones)
                    if (b != null) boneByName[b.name] = b;
        }

        foreach (var e in result.ellipsoids)
        {
            var go = new GameObject(e.name);
            var data = go.AddComponent<EllipsoidData>();
            data.radii = ToVec3(e.radii, Vector3.one * 0.1f);
            data.boneName = e.bone;

            Transform parent = transform;
            Vector3 localPos;
            Quaternion localRot;

            if (result.rigged)
            {
                if (e.bone != null && boneByName != null &&
                    boneByName.TryGetValue(e.bone, out var bone))
                    parent = bone;
                localPos = ToVec3(e.local_center, Vector3.zero);
                localRot = ToQuat(e.local_rotation);
            }
            else
            {
                localPos = ToVec3(e.center, Vector3.zero);
                localRot = ToQuat(e.rotation);
            }

            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localRotation = localRot;
        }
    }

    void ClearPrevious()
    {
        var stale = new List<EllipsoidData>();
        GetComponentsInChildren(true, stale);
        foreach (var d in stale)
            if (d != null && d.gameObject != gameObject)
            {
                if (Application.isPlaying) Destroy(d.gameObject);
                else DestroyImmediate(d.gameObject);
            }
    }

    // ── JSON serialization helpers (InvariantCulture, no JsonUtility) ────────

    static void AppendFloat(StringBuilder sb, float f) =>
        sb.Append(f.ToString("R", CultureInfo.InvariantCulture));

    static void AppendVec3(StringBuilder sb, float x, float y, float z)
    {
        sb.Append('[');
        AppendFloat(sb, x); sb.Append(',');
        AppendFloat(sb, y); sb.Append(',');
        AppendFloat(sb, z);
        sb.Append(']');
    }

    static void AppendVec4(StringBuilder sb, float x, float y, float z, float w)
    {
        sb.Append('[');
        AppendFloat(sb, x); sb.Append(',');
        AppendFloat(sb, y); sb.Append(',');
        AppendFloat(sb, z); sb.Append(',');
        AppendFloat(sb, w);
        sb.Append(']');
    }

    static string Escape(string s) =>
        s == null ? "" : s.Replace("\\", "\\\\").Replace("\"", "\\\"");

    static Vector3 ToVec3(float[] a, Vector3 fallback) =>
        (a != null && a.Length >= 3) ? new Vector3(a[0], a[1], a[2]) : fallback;

    static Quaternion ToQuat(float[] a) =>
        (a != null && a.Length >= 4)
            ? new Quaternion(a[0], a[1], a[2], a[3]) : Quaternion.identity;

    // ── Response DTOs (JsonUtility) ──────────────────────────────────────────

    [Serializable] class FitResp { public string job_id; }
    [Serializable]
    class StatusResp
    {
        public string job_id; public string state; public int step;
        public int total; public float loss; public int count; public string error;
    }
    [Serializable]
    class FitResult
    {
        public int version; public string coordinate_system;
        public string quaternion_convention; public bool rigged; public int count;
        public EllipsoidEntry[] ellipsoids;
    }
    [Serializable]
    class EllipsoidEntry
    {
        public string name; public string bone;
        public float[] center; public float[] local_center;
        public float[] radii;
        public float[] rotation; public float[] local_rotation;
    }
}
