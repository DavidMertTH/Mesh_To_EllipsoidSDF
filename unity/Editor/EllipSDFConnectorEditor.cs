// EllipSDFConnectorEditor.cs
// Must live in an Editor/ folder. Draws a "Fit Ellipsoids" button on the
// EllipSDFConnector inspector. The fit runs as a coroutine, so it needs Play
// mode — the button is disabled (with a hint) while in edit mode.

using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(EllipSDFConnector))]
public class EllipSDFConnectorEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        EditorGUILayout.Space();

        if (!Application.isPlaying)
        {
            EditorGUILayout.HelpBox(
                "Enter Play mode to run a fit (it uses coroutines / web requests).",
                MessageType.Info);
        }

        using (new EditorGUI.DisabledScope(!Application.isPlaying))
        {
            if (GUILayout.Button("Fit Ellipsoids", GUILayout.Height(30)))
                ((EllipSDFConnector)target).Fit();
        }
    }
}
