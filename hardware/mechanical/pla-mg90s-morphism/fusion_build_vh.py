"""
Fusion 360 script: build the VH snake (17 modules, new rib/coupler/caboose) as solid bodies
and run Fusion's interference analysis on it.

Needs assembly/parts_vh/*.stl + poses.json from export_fusion_vh.py.
Each mesh is converted to an exact faceted B-rep (MeshConvertFeatures) so Fusion's
interference tool can use it. One component per rigid body type, reused by occurrences.

run(...)                      -> new document "Slitherbot VH17", placed in the 'straight' pose
check_pose(pose_name)         -> move occurrences to a pose from poses.json and report interference
"""
import json
import os

import adsk.core
import adsk.fusion

PARTS = r"C:\Users\user\Desktop\dclab\robot\pla-mg90s-morphism\assembly\parts_vh"
MIN_VOLUME_MM3 = 0.5


def _poses():
    with open(os.path.join(PARTS, "poses.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _matrix(T):
    m = adsk.core.Matrix3D.create()
    flat = []
    for r in range(4):
        for c in range(4):
            v = T[r][c]
            flat.append(v / 10.0 if (c == 3 and r < 3) else v)   # mm -> cm
    m.setWithArray(flat)
    return m


def run(_context: str, new_document=True):
    app = adsk.core.Application.get()
    if new_document:
        app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    design = adsk.fusion.Design.cast(app.activeProduct)
    design.designType = adsk.fusion.DesignTypes.ParametricDesignType
    root = design.rootComponent
    data = _poses()
    mm = adsk.fusion.MeshUnits.MillimeterMeshUnit

    comps, first_occ = {}, {}
    for cname, parts in data["components"].items():
        occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        comp = occ.component
        comp.name = cname
        for p in parts:
            mb = comp.meshBodies.add(os.path.join(PARTS, p + ".stl"), mm).item(0)
            mb.name = p
            ci = comp.features.meshConvertFeatures.createInput([mb])
            ci.meshConvertMethodType = adsk.fusion.MeshConvertMethodTypes.FacetedMeshConvertMethodType
            comp.features.meshConvertFeatures.add(ci)
        for i in range(comp.bRepBodies.count):
            comp.bRepBodies.item(i).name = parts[i] if i < len(parts) else comp.bRepBodies.item(i).name
        for i in range(comp.meshBodies.count):
            comp.meshBodies.item(i).isLightBulbOn = False
        comps[cname], first_occ[cname] = comp, occ

    used = set()
    for pl in data["poses"]["straight"]:
        if pl["comp"] not in used:
            occ = first_occ[pl["comp"]]; used.add(pl["comp"])
            occ.transform2 = _matrix(pl["T"])
        else:
            occ = root.occurrences.addExistingComponent(comps[pl["comp"]], _matrix(pl["T"]))
        occ.attributes.add("snake", "name", pl["name"])
    for cname, occ in first_occ.items():
        if cname not in used:
            occ.deleteMe()
    app.activeViewport.fit()
    print("occurrences", root.occurrences.count,
          "| solids per component", {c: comps[c].bRepBodies.count for c in comps})


def _chain_occurrences(root):
    """Occurrences in chain order: base, m00 rib, m01 coupler, m01 rib, ..."""
    by_name = {}
    for i in range(root.occurrences.count):
        o = root.occurrences.item(i)
        a = o.attributes.itemByName("snake", "name")
        if a:
            by_name[a.value] = o
    order = sorted(by_name, key=lambda n: (n != "base", n[1:3] if n != "base" else "", "rib" in n))
    return by_name, order


def set_pose(pose_name):
    app = adsk.core.Application.get()
    root = adsk.fusion.Design.cast(app.activeProduct).rootComponent
    by_name, _ = _chain_occurrences(root)
    for pl in _poses()["poses"][pose_name]:
        by_name[pl["name"]].transform2 = _matrix(pl["T"])
    app.activeViewport.refresh()


def check_windows(start, stop, width=4):
    """Fusion interference on sliding windows of `width` consecutive bodies along the chain
    (covers every pair of bodies up to width-1 apart). Returns hits for windows [start, stop)."""
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    by_name, order = _chain_occurrences(root)
    hits, failed = [], []

    def analyse(entities):
        coll = adsk.core.ObjectCollection.create()
        for e in entities:
            coll.add(e)
        inp = design.createInterferenceInput(coll)
        inp.areCoincidentFacesIncluded = False
        return design.analyzeInterference(inp)

    def collect(res):
        for i in range(res.count):
            r = res.item(i)
            o1 = r.entityOne.assemblyContext; o2 = r.entityTwo.assemblyContext
            n1 = o1.attributes.itemByName("snake", "name").value if o1 else "?"
            n2 = o2.attributes.itemByName("snake", "name").value if o2 else "?"
            vol = r.interferenceBody.volume * 1000.0
            if n1 != n2 and vol > MIN_VOLUME_MM3:
                hits.append((n1, r.entityOne.name, n2, r.entityTwo.name, round(vol, 2)))

    def bodies(o):
        return [o.bRepBodies.item(i) for i in range(o.bRepBodies.count)]

    for w in range(start, min(stop, len(order) - width + 1)):
        names = order[w:w + width]
        res = analyse([by_name[n] for n in names])
        if res is not None:
            collect(res)
            continue
        # Fusion could not compute the window: fall back to body pairs across occurrences
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                for ba in bodies(by_name[names[a]]):
                    for bb in bodies(by_name[names[b]]):
                        if not ba.boundingBox.intersects(bb.boundingBox):
                            continue
                        r = analyse([ba, bb])
                        if r is None:
                            failed.append((names[a], ba.name, names[b], bb.name))
                        else:
                            collect(r)
    return sorted(set(hits)), sorted(set(failed)), len(order)


def check_all(pose_names=("straight", "gait_A30_headlift30", "coil_A55_lag90"), width=4,
              log=os.path.join(PARTS, "fusion_interference.jsonl")):
    """Run the windowed interference check for each pose; append one JSON line per window to `log`
    so progress survives if the caller (e.g. an MCP request) times out."""
    import time
    for pose in pose_names:
        set_pose(pose)
        root = adsk.fusion.Design.cast(adsk.core.Application.get().activeProduct).rootComponent
        _, order = _chain_occurrences(root)
        for w in range(len(order) - width + 1):
            t = time.time()
            hits, failed, _ = check_windows(w, w + 1, width)
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"pose": pose, "window": w, "bodies": order[w:w + width], "hits": hits,
                                     "failed": failed, "seconds": round(time.time() - t, 1)}) + "\n")
            adsk.doEvents()
    set_pose("straight")
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"done": True}) + "\n")


def check_pose(pose_name):
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    by_name = {}
    for i in range(root.occurrences.count):
        o = root.occurrences.item(i)
        a = o.attributes.itemByName("snake", "name")
        if a:
            by_name[a.value] = o
    for pl in _poses()["poses"][pose_name]:
        by_name[pl["name"]].transform2 = _matrix(pl["T"])
    coll = adsk.core.ObjectCollection.create()
    for o in by_name.values():
        coll.add(o)
    inp = design.createInterferenceInput(coll)
    inp.areCoincidentFacesIncluded = False
    res = design.analyzeInterference(inp)
    hits = []
    for i in range(res.count):
        r = res.item(i)
        def owner(e):
            occ = e.assemblyContext if hasattr(e, "assemblyContext") and e.assemblyContext else e
            a = occ.attributes.itemByName("snake", "name") if hasattr(occ, "attributes") else None
            return (a.value if a else occ.name), (e.name if hasattr(e, "name") else "")
        o1, b1 = owner(r.entityOne)
        o2, b2 = owner(r.entityTwo)
        vol = r.interferenceBody.volume * 1000.0   # cm3 -> mm3
        if o1 != o2 and vol > MIN_VOLUME_MM3:
            hits.append((o1, b1, o2, b2, round(vol, 2)))
    app.activeViewport.refresh()
    print(f"pose {pose_name}: {res.count} raw results, {len(hits)} between different bodies > {MIN_VOLUME_MM3} mm3")
    for h in hits:
        print("   ", h)
    return hits
