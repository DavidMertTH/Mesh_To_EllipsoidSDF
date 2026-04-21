// EllipsoidData.cs
// Place this file anywhere under Assets/Scripts/ (NOT in an Editor folder).
//
// Each imported ellipsoid becomes a child GameObject of its bone Transform.
// The GameObject's localPosition / localRotation encode the bone-local transform.
// This component stores the semi-axis lengths (radii) so you can query them at runtime.

using UnityEngine;

/// <summary>
/// Stores the semi-axis lengths of one fitted ellipsoid.
/// The GameObject's localPosition and localRotation encode the
/// bone-local center and orientation — set by EllipsoidImporter.
/// </summary>
public class EllipsoidData : MonoBehaviour
{
    [Tooltip("Semi-axis lengths (half-extents) along local X / Y / Z")]
    public Vector3 radii = Vector3.one * 0.1f;

    [Tooltip("Name of the bone this ellipsoid was assigned to (informational)")]
    public string boneName;

    // ── Runtime SDF query ────────────────────────────────────────────────────

    /// <summary>
    /// Signed distance from <paramref name="worldPoint"/> to this ellipsoid
    /// using the Inigo Quilez approximation.
    /// Negative = inside, positive = outside.
    /// </summary>
    public float SignedDistance(Vector3 worldPoint)
    {
        // Transform point into ellipsoid local space
        Vector3 p = transform.InverseTransformPoint(worldPoint);
        return SdfQuilez(p, radii);
    }

    /// <summary>Returns true if <paramref name="worldPoint"/> is inside the ellipsoid.</summary>
    public bool Contains(Vector3 worldPoint) => SignedDistance(worldPoint) <= 0f;

    // Inigo Quilez ellipsoid SDF approximation
    static float SdfQuilez(Vector3 p, Vector3 r)
    {
        float k0 = new Vector3(p.x / r.x, p.y / r.y, p.z / r.z).magnitude;
        float k1 = new Vector3(p.x / (r.x * r.x), p.y / (r.y * r.y), p.z / (r.z * r.z)).magnitude;
        return k0 * (k0 - 1f) / k1;
    }

    // ── Scene-view gizmos ────────────────────────────────────────────────────

    [Header("Gizmos")]
    [Tooltip("Draw gizmo even when not selected")]
    public bool drawAlways = false;

    void OnDrawGizmos()
    {
        if (drawAlways) DrawGizmo(new Color(0.25f, 0.85f, 0.45f, 0.25f));
    }

    void OnDrawGizmosSelected()
    {
        DrawGizmo(new Color(0.25f, 0.85f, 0.45f, 0.7f));
    }

    void DrawGizmo(Color color)
    {
        Gizmos.color = color;
        Matrix4x4 prev = Gizmos.matrix;
        // Scale a unit sphere to ellipsoid extents (diameter = radii * 2)
        Gizmos.matrix = Matrix4x4.TRS(transform.position, transform.rotation, radii * 2f);
        Gizmos.DrawWireSphere(Vector3.zero, 0.5f);
        Gizmos.matrix = prev;
    }
}
