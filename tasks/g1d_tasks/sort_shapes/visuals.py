"""Scene-only appearance overrides; the imported physics USD stays unchanged."""

from pxr import Gf, Sdf, UsdShade
import isaaclab.sim as sim_utils


def apply_robot_materials(env, env_ids):
    stage = sim_utils.get_current_stage()
    palette = {
        "shell": ((0.32, 0.36, 0.40), 0.35, 0.15),
        "motor": ((0.045, 0.055, 0.07), 0.4, 0.25),
        "rubber": ((0.018, 0.021, 0.026), 0.85, 0.0),
        "aluminium": ((0.40, 0.43, 0.46), 0.32, 0.6),
        "head": ((0.035, 0.045, 0.06), 0.28, 0.1),
    }
    materials = {}
    for name, (color, roughness, metallic) in palette.items():
        path = f"/World/G1D_Materials/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[name] = material
    count = 0
    for index in range(env.num_envs):
        robot = stage.GetPrimAtPath(f"/World/envs/env_{index}/Robot")
        for link in robot.GetChildren():
            visuals = link.GetChild("visuals")
            if not visuals.IsValid():
                continue
            name = link.GetName().lower()
            kind = "shell"
            if "wheel" in name:
                kind = "rubber"
            elif "head" in name or "logo" in name:
                kind = "head"
            elif "dex1" in name or name.startswith("lz_"):
                kind = "aluminium"
            elif "wrist" in name or "elbow" in name or "shoulder_roll" in name:
                kind = "motor"
            UsdShade.MaterialBindingAPI.Apply(visuals).Bind(
                materials[kind], bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            count += 1
    print(f"[G1-D] Applied body/motor/gripper materials to {count} visual groups", flush=True)
