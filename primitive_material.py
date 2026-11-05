"""Glossy dielectric appearance shared by GL and raymarched primitives.

The body retains its diffuse colour. Small neutral highlights provide gloss
without a broad metallic sheen or colour-tinted reflections.
"""

AMBIENT = 0.28
KEY_DIFFUSE = 0.62
FILL_DIFFUSE = 0.24
RIM_DIFFUSE = 0.12
DIFFUSE_CAP = 1.0
SPECULAR_POWER = 128.0
SPECULAR_STRENGTH = 0.82
FILL_SPECULAR_STRENGTH = 0.10
SHADER_NAME = "glossyPrimitive"

VERTEX_SHADER = """
uniform mat4 u_mvp;
uniform mat3 u_normal;
attribute vec4 a_position;
attribute vec3 a_normal;
attribute vec4 a_color;
varying vec4 v_color;
varying vec3 v_normal;
void main() {
    v_normal = normalize(u_normal * a_normal);
    v_color = a_color;
    gl_Position = u_mvp * a_position;
}
"""

FRAGMENT_SHADER = f"""
#ifdef GL_ES
precision mediump float;
#endif
varying vec4 v_color;
varying vec3 v_normal;
void main() {{
    vec3 n = normalize(v_normal);
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 nf = dot(n, V) < 0.0 ? -n : n;
    vec3 key = normalize(vec3(0.4, -0.5, 0.85));
    vec3 fill = normalize(vec3(-0.6, 0.35, 0.45));
    vec3 rim = normalize(vec3(0.15, 0.85, 0.35));
    float diffuse = {AMBIENT};
    diffuse += max(dot(nf, key), 0.0) * {KEY_DIFFUSE};
    diffuse += max(dot(nf, fill), 0.0) * {FILL_DIFFUSE};
    diffuse += max(dot(nf, rim), 0.0) * {RIM_DIFFUSE};
    vec3 rgb = v_color.rgb * min(diffuse, {DIFFUSE_CAP});
    float specular = pow(max(dot(nf, normalize(key + V)), 0.0), {SPECULAR_POWER}) * {SPECULAR_STRENGTH};
    specular += pow(max(dot(nf, normalize(fill + V)), 0.0), {SPECULAR_POWER}) * {FILL_SPECULAR_STRENGTH};
    gl_FragColor = vec4(min(rgb + vec3(specular), vec3(1.0)), v_color.a);
}}
"""
