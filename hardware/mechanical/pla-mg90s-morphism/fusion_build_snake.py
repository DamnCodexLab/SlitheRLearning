"""
Fusion 360 script: build the 4-node Slitherbot body as a jointed assembly.

Run from Fusion (Utilities > Scripts and Add-Ins) or via the Fusion MCP.
Needs the part STLs exported by assemble_snake.py into assembly/parts/.

Each moving node is one rigid component:
  base        : caboose + tail shell + its wheels + first coupler + MG90S (grounded)
  link_mid    : rib + wire guide + 2 wheels + servo horn
                + the NEXT coupler + MG90S, bolted to the rib's rear leg (x + 48.9 mm)
  link_head   : rib parts + the head shell (last node)
Chain:  base --servo_0--> link_0 --servo_1--> link_1 --servo_2--> link_2 --servo_3--> link_3
Each servo_k is a revolute joint about the design Y axis (up) through x=5, z=0
of that node.  Design frame: x = snake axis, y = up, mm.
"""
import os
import adsk.core
import adsk.fusion

PARTS = r"C:\Users\user\Desktop\dclab\robot\pla-mg90s-morphism\assembly\parts"
N_NODES = 4
PITCH_CM = 4.89
AXIS_X_CM = 0.5
NEW_DOCUMENT = True   # False: clear and reuse the active design
LIMIT_RAD = 1.05      # ~ +-60 deg

RIB_PARTS = ["rib", "wire_guide", "wheel_R", "wheel_L", "mg90s_horn"]
CPL_PARTS = ["coupler", "mg90s_body"]
BASE_PARTS = CPL_PARTS + ["caboose", "caboose_wheels", "tail_wire_guide", "tail"]
HEAD_PARTS = ["head_bottom_mounted", "head_top_mounted"]  # pre-placed on the front rib leg


def run(_context: str):
    app = adsk.core.Application.get()
    if NEW_DOCUMENT:
        app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    design = adsk.fusion.Design.cast(app.activeProduct)
    design.designType = adsk.fusion.DesignTypes.ParametricDesignType
    root = design.rootComponent
    for i in reversed(range(root.occurrences.count)):
        root.occurrences.item(i).deleteMe()
    mm = adsk.fusion.MeshUnits.MillimeterMeshUnit

    def move(x_cm=0.0):
        m = adsk.core.Matrix3D.create()
        m.translation = adsk.core.Vector3D.create(x_cm, 0, 0)
        return m

    def import_mesh(comp, name):
        comp.meshBodies.add(os.path.join(PARTS, name + ".stl"), mm).item(0).name = name

    def axis_points(comp, xs):
        sk = comp.sketches.add(comp.xYConstructionPlane)
        sk.name = "servo_axes"
        return [sk.sketchPoints.add(adsk.core.Point3D.create(x, 0, 0)) for x in xs]

    def new_comp(name, parts, next_coupler):
        occ = root.occurrences.addNewComponent(move())
        comp = occ.component
        comp.name = name
        for p in parts:
            import_mesh(comp, p)
        if next_coupler:
            for p in CPL_PARTS:
                import_mesh(comp, p + "_next")  # pre-shifted by 48.9 mm
        xs = [AXIS_X_CM] + ([AXIS_X_CM + PITCH_CM] if next_coupler else [])
        return occ, axis_points(comp, xs)

    base, base_pts = new_comp("base_caboose", BASE_PARTS, False)
    mid0, mid_pts = new_comp("link_mid", RIB_PARTS, True)
    end0, end_pts = new_comp("link_head", RIB_PARTS + HEAD_PARTS, False)
    base.isGrounded = True

    links, pts = [], []
    for k in range(N_NODES):
        last = k == N_NODES - 1
        if k == 0:
            occ = mid0 if not last else end0
        elif last:
            occ = end0
            occ.transform2 = move(PITCH_CM * k)
        else:
            occ = root.occurrences.addExistingComponent(mid0.component, move(PITCH_CM * k))
        links.append(occ)
        pts.append(end_pts if last else mid_pts)
    if N_NODES > 1 and end0 is not links[-1]:
        end0.deleteMe()

    joints = root.joints
    for k in range(N_NODES):
        if k == 0:
            g_parent = adsk.fusion.JointGeometry.createByPoint(base_pts[0].createForAssemblyContext(base))
        else:
            g_parent = adsk.fusion.JointGeometry.createByPoint(pts[k - 1][1].createForAssemblyContext(links[k - 1]))
        g_child = adsk.fusion.JointGeometry.createByPoint(pts[k][0].createForAssemblyContext(links[k]))
        ji = joints.createInput(g_child, g_parent)
        ji.setAsRevoluteJointMotion(adsk.fusion.JointDirections.YAxisJointDirection)
        j = joints.add(ji)
        j.name = f"servo_{k}"
        lim = j.jointMotion.rotationLimits
        lim.isMinimumValueEnabled = True; lim.minimumValue = -LIMIT_RAD
        lim.isMaximumValueEnabled = True; lim.maximumValue = LIMIT_RAD

    app.activeViewport.fit()
    print("links:", len(links), "joints:", joints.count)
