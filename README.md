# Mesh_To_EllipsoidSDF

## Unity primitive pipeline

The Unity bridge supports two fitted primitive types:

- `ellipsoid`
- `superquadric` with per-primitive shape exponents `[epsilon1, epsilon2]`

Protocol version 4 carries `primitive_type` and `shape_exponents` through base
fitting, fixed-population pose fitting, synthetic pose batches, bone-local
storage, morph targets, previews, and JSON export. Version 2/3 data without the
new fields remains valid and is interpreted as an ellipsoid with exponents
`[1, 1]`.

> **NOT IMPLEMENTED:** Superquadric collision detection/response in Unity.
> Superquadrics can be fitted, previewed, stored, and morphed, but they are
> deliberately not submitted to the existing ellipsoid collision pipeline.
