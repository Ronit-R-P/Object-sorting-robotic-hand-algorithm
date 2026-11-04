import sys, os, json, time, numpy as np, mujoco, mujoco.viewer
try:
    import cv2
except ImportError:
    cv2 = None

# =============================== SETTINGS ===============================
XML_PATH = r"F:\robotic_arm\Mujoco Training\whole arm\all_objects.xml"   # the ONE xml that holds every object
POSE_DIR = r"F:\video-to-hand_IRC"
HAND     = "hand_root"
ARM_ACT  = ["Link1_act","Link2_act","Link3_act","Link4_act","Link5_act"]
FING_ACT = ["Index_MCP_act","Index_PIP_DIP_act","Middle_MCP_act","Middle_PIP_DIP_act",
            "Ring_MCP_act","Ring_PIP_DIP_act","Pinky_MCP_act","Pinky_PIP_DIP_act",
            "Thumb_Knuckle_act","Thumb_MCP_act"]
THUMB_ROT, THUMB_CLOSE = "Thumb_Knuckle_act", "Thumb_MCP_act"
GRAV_COMP  = True
PICK_ORDER = "base"     # "base" = closest to the arm first | "bin" = closest to its own bin first
CLEARANCE  = 0.04       # gap (m) kept between a carried object and the highest other object
REF_BIN_XY = None       # where bin_red was when the bin poses were recorded. None = where bin_red is now ((0.932, 0.268) in the cylinder file)
ONLY       = None       # for testing one object: e.g. ["test_mug"]. The others stay in the scene as obstacles. None = pick everything

# Which skill handles which body: by NAME. "mug" in the name -> mug, "temple" in the name -> temple, anything else -> cylinder.
# Pose files per skill:  cylinder: pose_pre.json ...   mug: pose_cube_pre.json ...   temple: pose_temple_pre.json ...
POSE_PREFIX = {"cylinder": "pose_", "mug": "pose_cube_", "temple": "pose_temple_"}

# Every tuned value from the three separate files, unchanged.
CFG = {
    "cylinder": dict(slow=1.0, sleep=False,
                     SQUEEZE=0.15, LIFT=0.45, THUMB_MCP_CLOSE=0.15, THUMB_ROT_OPEN="max", THUMB_FLEX_OPEN="max",
                     CLOSE_STAGES=[(0.6, 0.0), (1.0, 0.0), (1.0, 1.0)], STAGE_T=0.4,
                     REF_XY=(0.932, -0.082)),            # only used if pose_pre.json has no "cyl" entry
    "mug":      dict(slow=1, sleep=False,
                     OFFSET_DEG=-2.5, STOP_FRAC=0.85, EXTRA_IN_DEG=-3.5, LIFT_CM=18,
                     DROP_WRIST_EXTRA_DEG=-50, RETREAT_LIFT_CM=10, RADIAL_FIX=True),
    "temple":   dict(slow=1.5, sleep=True,
                     IN_CM=3, LIFT_CM=12, DROP_WRIST_EXTRA_DEG=-65, RETREAT_LIFT_CM=10, RADIAL_FIX=True),
}

# =============================== MODEL ===============================
m = mujoco.MjModel.from_xml_path(XML_PATH); d = mujoco.MjData(m)
aid  = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
A, F = [aid(n) for n in ARM_ACT], [aid(n) for n in FING_ACT]
hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HAND)
arm_j   = [int(m.actuator_trnid[a, 0]) for a in A]
arm_dof = [int(m.jnt_dofadr[j]) for j in arm_j]
arm_q   = [int(m.jnt_qposadr[j]) for j in arm_j]
lo, hi  = m.actuator_ctrlrange[A, 0], m.actuator_ctrlrange[A, 1]
iR, iC  = FING_ACT.index(THUMB_ROT), FING_ACT.index(THUMB_CLOSE)
i4      = [i for i in range(len(F)) if i not in (iR, iC)]

def kind_of(name):
    n = name.lower()
    return "mug" if "mug" in n else "temple" if "temple" in n else "cylinder"

objs = []                                        # every free-joint body is an object
for j in range(m.njnt):
    if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE): continue
    b = int(m.jnt_bodyid[j]); g0 = int(m.body_geomadr[b])
    r_, g_ = float(m.geom_rgba[g0][0]), float(m.geom_rgba[g0][1])
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b and (m.geom_contype[g] or m.geom_conaffinity[g])]
    objs.append(dict(name=name, body=b, adr=int(m.jnt_qposadr[j]), geoms=geoms,
                     color="yellow" if g_ > 0.5 * r_ else "red", kind=kind_of(name)))
if not objs: sys.exit("No free-joint objects found in the XML.")
obj_bodies = {o["body"] for o in objs}

G_TABLE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
BIN     = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"bin_{n}") for n in ("red", "yellow")}
if G_TABLE < 0 or min(BIN.values()) < 0: sys.exit('XML needs a geom "table" and bodies "bin_red", "bin_yellow".')
TABLE_TOP = float(m.geom_pos[G_TABLE][2] + m.geom_size[G_TABLE][2])

if GRAV_COMP:                                    # arm and fixtures only: NEVER the objects
    for b in range(1, m.nbody):
        if b not in obj_bodies: m.body_gravcomp[b] = 1.0
dt10 = 10 * m.opt.timestep

# optional live camera window (only if the XML has a "third_person_cam")
CAM = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "third_person_cam")
ren = None
if cv2 is not None and CAM >= 0:
    ren = mujoco.Renderer(m, height=360, width=480)
    cv2.namedWindow("3rd Person Vision Feed", cv2.WINDOW_NORMAL); cv2.resizeWindow("3rd Person Vision Feed", 480, 360)

class S: pass                                    # scene constants and loaded skills, filled by setup()
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
def ang(p):     return np.arctan2(p[1] - S.pivot[1], p[0] - S.pivot[0])
def cpos(o):    return d.xpos[o["body"]].copy()
def bin_xy(o):  return d.xpos[BIN[o["color"]]][:2].copy()

# =============================== HELPERS ===============================
def step(n):
    for _ in range(n): mujoco.mj_step(m, d)

def show():
    if ren is None: return
    ren.update_scene(d, camera="third_person_cam")
    cv2.imshow("3rd Person Vision Feed", cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)); cv2.waitKey(1)

def move(v, arm=None, fing=None, T=2.0):
    c0, n = d.ctrl.copy(), max(1, int(T * S.slow / dt10))
    for k in range(1, n + 1):
        s = k / n; s = s * s * (3 - 2 * s)
        if arm  is not None: d.ctrl[A] = (1 - s) * c0[A] + s * arm
        if fing is not None: d.ctrl[F] = (1 - s) * c0[F] + s * fing
        step(10); v.sync()
        if k % 4 == 0: show()
        if S.sleep: time.sleep(dt10)

def teleport(ctrl):
    for a in A + F: d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]] = ctrl[a]
    d.qvel[:] = 0; d.ctrl[:] = ctrl; mujoco.mj_forward(m, d)

def ik(target, rel, q0, iters=300):
    """Position-only IK of the point 'rel' (hand frame). rel = zeros(3) means the hand origin."""
    dk = mujoco.MjData(m); dk.qpos[:] = d.qpos
    q, qs, jp = np.array(q0, float), np.array(q0, float), np.zeros((3, m.nv))
    for _ in range(iters):
        for i, qa in enumerate(arm_q): dk.qpos[qa] = q[i]
        mujoco.mj_kinematics(m, dk); mujoco.mj_comPos(m, dk)
        R = dk.xmat[hand].reshape(3, 3); p = dk.xpos[hand] + R @ rel
        e = target - p
        if np.linalg.norm(e) < 5e-4: break
        mujoco.mj_jac(m, dk, jp, None, p, hand)
        J  = jp[:, arm_dof]
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(3))
        dq = Jp @ e + (np.eye(len(q)) - Jp @ J) @ (qs - q) * 0.2
        q  = np.clip(q + np.clip(dq, -0.1, 0.1), lo, hi)
    return q, float(np.linalg.norm(e))

def hand_origin(q):
    """Hand origin (world) for arm joint values q, other joints as they are now."""
    dk = mujoco.MjData(m); dk.qpos[:] = d.qpos
    for i, qa in enumerate(arm_q): dk.qpos[qa] = q[i]
    mujoco.mj_kinematics(m, dk)
    return dk.xpos[hand].copy()

def lift_hand(v, cm, steps=6):
    """Straight-up lift of the hand origin by IK (fingers untouched). Returns the last IK error (m)."""
    p0 = d.xpos[hand].copy(); q = d.ctrl[A].copy(); e = 0.0
    for k in range(1, steps + 1):
        q, e = ik(p0 + np.array([0, 0, cm / 100 * k / steps]), np.zeros(3), q)
        move(v, arm=q, T=0.5)
    return e

def geom_zext(g):
    """How far geom g reaches above/below its centre in world z (exact for primitives)."""
    R = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g]; t = int(m.geom_type[g]); GT = mujoco.mjtGeom
    if t == int(GT.mjGEOM_BOX):       return float(np.abs(R[2]) @ s)
    if t == int(GT.mjGEOM_SPHERE):    return float(s[0])
    if t == int(GT.mjGEOM_CAPSULE):   return float(s[1] * abs(R[2, 2]) + s[0])
    if t == int(GT.mjGEOM_CYLINDER):  return float(s[1] * abs(R[2, 2]) + s[0] * np.sqrt(max(0.0, 1 - R[2, 2] ** 2)))
    if t == int(GT.mjGEOM_ELLIPSOID): return float(np.linalg.norm(R[2] * s))
    return float(m.geom_rbound[g])

def extent(o):
    """(lowest z, highest z) of an object, from its colliding geoms at the current pose."""
    zs = [(d.geom_xpos[g][2] - geom_zext(g), d.geom_xpos[g][2] + geom_zext(g)) for g in o["geoms"]]
    return min(a for a, _ in zs), max(b for _, b in zs)

def transit(v, posture_arm):
    """Rise in place: go to a high arm posture while keeping the current Link1 angle."""
    q = np.array(posture_arm, float).copy(); q[0] = d.ctrl[A][0]
    if np.max(np.abs(q - d.ctrl[A])) > 0.02: move(v, arm=q, T=2.5)

def ensure_clear(v, o, others):
    """Before swinging round: lift extra if the carried object would pass through any other object."""
    if not others: return
    need = max(extent(x)[1] for x in others) + CLEARANCE - extent(o)[0]
    if need > 0.005:
        e = lift_hand(v, need * 100)
        print(f"  extra lift {need*100:.0f} cm to clear the other objects: IK error {e*1000:.0f} mm")

def radial_shift(q, obj_xy, dr):
    """Approximate fix for an object nearer/farther than recorded (position-only IK, height kept)."""
    u = (obj_xy - S.pivot) / np.linalg.norm(obj_xy - S.pivot)
    q2, e = ik(hand_origin(q) + np.r_[u * dr, 0.0], np.zeros(3), q)
    print(f"  radial correction {dr*100:+.1f} cm: IK error {e*1000:.0f} mm")
    return q2

def report(o):
    bxy, p = bin_xy(o), cpos(o)
    inside = abs(p[0] - bxy[0]) < 0.14 and abs(p[1] - bxy[1]) < 0.14 and p[2] < 0.2
    print(f"  {o['name']} -> bin_{o['color']}: final position {p.round(3)}, inside bin = {inside}")

# =============================== SETUP ===============================
def load_kind(kind):
    C = CFG[kind]
    def J(tag, req=True):
        p = rf"{POSE_DIR}\{POSE_PREFIX[kind]}{tag}.json"
        if not os.path.exists(p):
            if req: sys.exit(f"Missing {p}  (needed for the {kind} skill, record it with its capture script)")
            return None
        return json.load(open(p))
    P, G = J("pre"), J("grasp")
    sk = dict(pre=np.array(P["ctrl"]), gr=np.array(G["ctrl"]),
              above=np.array(J("bin_above")["ctrl"]), drop=np.array(J("bin_drop")["ctrl"]))
    R = J("retreat", req=(kind == "cylinder"))
    sk["retreat"] = np.array(R["ctrl"]) if R else None
    if kind == "cylinder":
        sk["rel_pre"], sk["rel_gr"] = np.array(P["rel"]), np.array(G["rel"])
        sk["ref_p"] = np.array(P.get("cyl", [*C["REF_XY"], 0])[:2]); sk["ref_g"] = np.array(G.get("cyl", [*C["REF_XY"], 0])[:2])
        pre, gr = sk["pre"], sk["gr"]
        pf, gf = pre[F].copy(), gr[F]
        lo_f, hi_f = m.actuator_ctrlrange[F, 0], m.actuator_ctrlrange[F, 1]
        OPEN = np.where(gf > pf, lo_f, hi_f); pf[i4] = OPEN[i4]
        res = lambda o, i, df: lo_f[i] if o == "min" else hi_f[i] if o == "max" else df if o is None else float(o)
        pf[iR] = res(C["THUMB_ROT_OPEN"], iR, pf[iR]); pf[iC] = res(C["THUMB_FLEX_OPEN"], iC, pf[iC])
        sq = np.clip(gf + C["SQUEEZE"] * np.sign(gf - pf), lo_f, hi_f)
        def stage(a4, aC):
            t = pf.copy()
            t[i4] = pf[i4] + a4 * (gf[i4] - pf[i4])
            t[iR] = gf[iR]
            t[iC] = pf[iC] + aC * (C["THUMB_MCP_CLOSE"] * (hi_f[iC] - lo_f[iC])) * np.sign(gf[iC] - pf[iC])
            return t
        sk["pf"], sk["stage"], sk["thumb_open"] = pf, stage, stage(0.0, 0.0)
        sq = sq.copy(); sq[iC] = stage(1.0, 1.0)[iC]; sk["sq"] = sq
    else:
        sk["ref"] = np.array(P["cyl"][:2], float) if "cyl" in P else None   # where the object was when `pre` was recorded
    return sk

def setup():
    hp = rf"{POSE_DIR}\pose_home.json"
    S.home = np.array(json.load(open(hp))["ctrl"]) if os.path.exists(hp) else \
             np.clip(np.zeros(m.nu), m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
    mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    S.pivot = d.xanchor[arm_j[0]][:2].copy()
    S.sgn = 1.0 if d.xaxis[arm_j[0]][2] > 0 else -1.0
    S.ref_bin = np.array(REF_BIN_XY, float) if REF_BIN_XY is not None else d.xpos[BIN["red"]][:2].copy()

    for o in objs:                                # rest every object ON the table: keep its XML x, y and orientation
        d.qpos[o["adr"] + 2] += TABLE_TOP + 0.002 - extent(o)[0]
    mujoco.mj_forward(m, d)

    S.sk = {k: load_kind(k) for k in sorted({o["kind"] for o in objs})}
    S.slow, S.sleep = 1.0, False
    teleport(S.home); step(300)                   # arm at home, objects settle
    print(f"arm pivot {S.pivot.round(3)}, bin_red reference {S.ref_bin.round(3)}")
    for o in objs:
        p = cpos(o)
        print(f"  {o['name']:14s} {o['kind']:9s} {o['color']:6s} at {p.round(3)}, radius {np.linalg.norm(p[:2]-S.pivot):.3f} m -> bin_{o['color']}")

# =============================== CYLINDER SKILL ===============================
def plan_cylinder(c, others):
    sk, C = S.sk["cylinder"], CFG["cylinder"]
    p0 = cpos(c); cxy = p0[:2]
    pre = sk["pre"].copy()
    pre[A[0]] += S.sgn * wrap(ang(cxy) - ang(sk["ref_p"]))
    q_high, _ = ik(p0 + np.array([0, 0, 0.12]), sk["rel_pre"], pre[A])
    q_high[4] = np.clip(q_high[4] + 0.35, lo[4], hi[4])
    q_pre, e1 = ik(p0 + np.array([0, 0, 0.02]), sk["rel_pre"], q_high)
    radial = (cxy - S.pivot) / np.linalg.norm(cxy - S.pivot)
    deep = p0.copy(); deep[:2] += radial * 0.01
    q_gr, e2 = ik(deep, sk["rel_gr"], q_pre)
    need = 0.0                                    # lift needed so the held cylinder clears the others
    if others: need = max(0.0, max(extent(x)[1] for x in others) + CLEARANCE - extent(c)[0])
    lift_ok, e3 = True, 0.0
    if need > C["LIFT"]:
        _, e3 = ik(deep + np.array([0, 0, need]), sk["rel_gr"], q_gr)
        lift_ok = e3 < 0.015
    return dict(q_high=q_high, q_pre=q_pre, q_gr=q_gr, e1=e1, e2=e2, need=need, e3=e3, lift_ok=lift_ok, z0=float(p0[2]))

def pick_cylinder(v, c, pl, others):
    sk, C = S.sk["cylinder"], CFG["cylinder"]
    pf, sq, thumb_open, stage = sk["pf"], sk["sq"], sk["thumb_open"], sk["stage"]
    print(f"  IK errors: pre {pl['e1']*1000:.1f} mm, grasp {pl['e2']*1000:.1f} mm")
    if max(pl["e1"], pl["e2"]) > 0.005: print("  WARNING: target out of reach")
    transit(v, sk["retreat"][A])                  # raised posture, rise in place
    q_turn = d.ctrl[A].copy(); q_turn[0] = pl["q_high"][0]
    move(v, arm=q_turn, fing=pf, T=2.0)           # turn at height (only Link1 changes)
    move(v, arm=pl["q_high"], fing=pf, T=2.0)     # come down onto the cylinder
    move(v, arm=pl["q_pre"], fing=pf, T=1.0)
    move(v, fing=thumb_open, T=0.8)
    move(v, arm=pl["q_gr"], T=1.2)
    for a4, aC in C["CLOSE_STAGES"]: move(v, fing=stage(a4, aC), T=C["STAGE_T"])
    move(v, fing=sq, T=0.8)
    print(f"  grasp: contacts {n_contacts(c)}")

    LIFT = C["LIFT"]
    pg = d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ sk["rel_gr"]
    q = pl["q_gr"]
    for k in range(1, 6):
        q, _ = ik(pg + np.array([0, 0, LIFT * k / 5]), sk["rel_gr"], q)
        move(v, arm=q, T=0.3)
    if pl["need"] > LIFT:                         # extra height to clear the objects still on the table
        for k in range(1, 4):
            q, _ = ik(pg + np.array([0, 0, LIFT + (pl["need"] - LIFT) * k / 3]), sk["rel_gr"], q)
            move(v, arm=q, T=0.5)
    move(v, T=0.8)
    print(f"  lifted {(cpos(c)[2] - pl['z0']) * 100:.1f} cm (planned {max(pl['need'], LIFT) * 100:.0f} cm), contacts {n_contacts(c)}")

    bxy = bin_xy(c)
    dth = S.sgn * wrap(ang(bxy) - ang(S.ref_bin))
    qa, qd, qr = sk["above"][A].copy(), sk["drop"][A].copy(), sk["retreat"][A].copy()
    for qq in (qa, qd, qr): qq[0] = np.clip(qq[0] + dth, lo[0], hi[0])
    q_swing = q.copy(); q_swing[0] = qa[0]
    move(v, arm=q_swing, fing=sq, T=2.5)          # swing round at carry height
    move(v, arm=qa, fing=sq, T=2.0)
    move(v, arm=qd, fing=sq, T=1.2)
    move(v, fing=pf, T=0.8)                       # release
    move(v, arm=qr, fing=pf, T=1.5)
    move(v, T=1.0)
    report(c)

def n_contacts(o):
    n = 0
    for i in range(d.ncon):
        b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
        n += (o["body"] in (b1, b2)) and (0 not in (b1, b2))
    return n

# =============================== MUG + TEMPLE (joint-pose skills, made relative) ===============================
def locate(o, sk):
    """Where is the object compared with where its poses were recorded? Returns (turn round the base, radial change)."""
    p = cpos(o)
    ref = sk["ref"]
    if ref is None:
        ref = p[:2].copy(); print("  WARNING: pose_pre has no 'cyl' entry, assuming the object has not moved since it was recorded")
    phi = wrap(ang(p[:2]) - ang(ref))
    dr = np.linalg.norm(p[:2] - S.pivot) - np.linalg.norm(ref - S.pivot)
    print(f"  recorded at {ref.round(3)} | turn round the base {np.degrees(phi):+.1f} deg | nearer/farther {dr*100:+.1f} cm")
    print(f"  (to keep the object's orientation as recorded, add {phi:.3f} rad to its euler z in the XML)")
    if abs(dr) > 0.04: print("  WARNING: more than 4 cm off the recorded radius, expect a rougher grasp")
    return phi, dr

def carry_to_bin(v, o, sk):
    """Swing at height to the bin and settle into the recorded bin_above pose. Returns (drop pose, retreat pose or None)."""
    dth = S.sgn * wrap(ang(bin_xy(o)) - ang(S.ref_bin))
    qa, qd = sk["above"][A].copy(), sk["drop"][A].copy()
    qr = sk["retreat"][A].copy() if sk["retreat"] is not None else None
    for q_ in (qa, qd, qr):
        if q_ is not None: q_[0] = np.clip(q_[0] + dth, lo[0], hi[0])
    q_swing = d.ctrl[A].copy(); q_swing[0] = qa[0]
    move(v, arm=q_swing, fing=sk["above"][F], T=2.5)   # turn Link1 only, at height
    move(v, arm=qa, fing=sk["above"][F], T=2.0)        # then the other joints settle into the pose
    return qd, qr

def retreat_from_bin(v, sk, qr, C):
    if qr is not None: move(v, arm=qr, fing=sk["retreat"][F], T=2.0)
    else:
        print("  no retreat pose recorded, lifting the hand instead")
        lift_hand(v, C["RETREAT_LIFT_CM"])
    move(v, T=1.0)

def pick_mug(v, c, pl, others):
    sk, C = S.sk["mug"], CFG["mug"]
    phi, dr = locate(c, sk)
    q_pre = sk["pre"][A].copy(); q_pre[0] = np.clip(q_pre[0] + S.sgn * phi, lo[0], hi[0])   # ONLY Link1 turns: the recorded posture is kept
    dg = sk["gr"][A][0] - sk["pre"][A][0]                                                    # recorded Link1 slide pre -> grasp
    if C["RADIAL_FIX"] and abs(dr) > 0.01: q_pre = radial_shift(q_pre, cpos(c)[:2], dr)

    transit(v, S.home[A])
    # 1. APPROACH: turn at height beside the handle -> lower -> slide into the pre pose (Link1 only)
    q_side = q_pre.copy(); q_side[0] = np.clip(q_side[0] + np.radians(C["OFFSET_DEG"]), lo[0], hi[0])
    q_stage = S.home.copy()[A]; q_stage[0] = q_side[0]
    q_mid = q_stage + C["STOP_FRAC"] * (q_side - q_stage)
    move(v, arm=q_stage, fing=sk["pre"][F], T=1.5)
    move(v, arm=q_mid,   fing=sk["pre"][F], T=1.5)
    move(v, arm=q_side,  fing=sk["pre"][F], T=1.5)
    move(v, arm=q_pre,   fing=sk["pre"][F], T=1.5)
    # 2. GRASP: Link1 only slides in (same slide as recorded), then the fingers curl
    dirn = np.sign(dg) or 1.0
    q_in = d.ctrl[A].copy()
    q_in[0] = np.clip(q_pre[0] + dg + dirn * np.radians(C["EXTRA_IN_DEG"]), lo[0], hi[0])
    move(v, arm=q_in, T=1.5)
    move(v, fing=sk["gr"][F], T=1.5)
    # 3. LIFT straight up, fingers closed
    e = lift_hand(v, C["LIFT_CM"])
    print(f"  lift {C['LIFT_CM']} cm: IK error {e*1000:.0f} mm | height {cpos(c)[2]:.3f} m")
    ensure_clear(v, c, others)
    # 4. CARRY
    qd, qr = carry_to_bin(v, c, sk)
    # 5. DROP: wrist tips down first (fingers still closed), then the fingers open
    q_w = qd.copy()
    wdir = np.sign(qd[4] - d.ctrl[A][4]) or 1.0
    q_w[4] = np.clip(q_w[4] + wdir * np.radians(C["DROP_WRIST_EXTRA_DEG"]), lo[4], hi[4])
    move(v, arm=q_w, T=1.5)
    move(v, fing=sk["drop"][F], T=1.0)
    move(v, T=1.0)
    # 6. RETREAT
    retreat_from_bin(v, sk, qr, C)
    report(c)

def pick_temple(v, c, pl, others):
    sk, C = S.sk["temple"], CFG["temple"]
    phi, dr = locate(c, sk)
    q_pre = sk["pre"][A].copy(); q_pre[0] = np.clip(q_pre[0] + S.sgn * phi, lo[0], hi[0])
    q_gr  = sk["gr"][A].copy();  q_gr[0]  = np.clip(q_gr[0]  + S.sgn * phi, lo[0], hi[0])
    if C["RADIAL_FIX"] and abs(dr) > 0.01:
        q_pre = radial_shift(q_pre, cpos(c)[:2], dr); q_gr = radial_shift(q_gr, cpos(c)[:2], dr)

    transit(v, S.home[A])
    # 1. APPROACH: turn Link1 only at height -> lower onto the pre pose
    q_stage = S.home.copy()[A]; q_stage[0] = q_pre[0]
    move(v, arm=q_stage, fing=sk["pre"][F], T=1.5)
    move(v, arm=q_pre,   fing=sk["pre"][F], T=1.5)
    # 2. PUSH IN horizontally towards the object (direction measured now), fingers to the grasp values, then the recorded rise
    if C["IN_CM"] != 0:
        dirv = cpos(c)[:2] - d.xpos[hand][:2]; dirv = dirv / np.linalg.norm(dirv)
        p0 = d.xpos[hand].copy(); q = d.ctrl[A].copy()
        for k in range(1, 5):
            q, e = ik(p0 + np.r_[dirv * C["IN_CM"] / 100 * k / 4, 0.0], np.zeros(3), q)
            move(v, arm=q, T=0.6)
        print(f"  push-in {C['IN_CM']} cm: IK error {e*1000:.0f} mm")
    move(v, fing=sk["gr"][F], T=1.5)
    move(v, arm=q_gr, T=3.0)
    # 3. LIFT
    if C["LIFT_CM"] > 0:
        e = lift_hand(v, C["LIFT_CM"])
        print(f"  lift: IK error {e*1000:.0f} mm | height {cpos(c)[2]:.3f} m")
    ensure_clear(v, c, others)
    # 4. CARRY
    qd, qr = carry_to_bin(v, c, sk)
    # 5. DROP
    move(v, arm=qd, T=1.5)
    move(v, fing=sk["drop"][F], T=1.0)
    move(v, T=1.0)
    # 6. RETREAT
    retreat_from_bin(v, sk, qr, C)
    report(c)

# =============================== MAIN ===============================
def choose_next(remaining):
    if PICK_ORDER == "base":
        c = min(remaining, key=lambda x: np.linalg.norm(cpos(x)[:2] - S.pivot))
        if c["kind"] != "cylinder": return c, None
        pl = plan_cylinder(c, [o for o in remaining if o is not c])
        if pl["lift_ok"]: return c, pl
        print(f"WARNING: cannot lift the {c['name']} high enough to clear the others (IK error {pl['e3']*1000:.0f} mm). Switching to bin order.")
    c = min(remaining, key=lambda x: np.linalg.norm(cpos(x)[:2] - bin_xy(x)))
    return c, (plan_cylinder(c, []) if c["kind"] == "cylinder" else None)

SKILL = {"cylinder": pick_cylinder, "mug": pick_mug, "temple": pick_temple}

def main():
    setup()
    todo = [o for o in objs if ONLY is None or o["name"] in ONLY]
    if not todo: sys.exit(f"ONLY = {ONLY} matches no object. Objects in the XML: {[o['name'] for o in objs]}")
    with mujoco.viewer.launch_passive(m, d) as v:
        v.sync(); time.sleep(0.5)
        remaining, n = list(todo), 0
        while remaining and v.is_running():
            c, pl = choose_next(remaining)
            remaining.remove(c); n += 1
            C = CFG[c["kind"]]; S.slow, S.sleep = C["slow"], C["sleep"]
            print(f"\n[{n}/{len(todo)}] {c['name']} ({c['kind']}, {c['color']}) at {cpos(c).round(3)}")
            SKILL[c["kind"]](v, c, pl, [o for o in objs if o is not c and o in remaining])
        print("\nDone." if not remaining else "\nStopped.")
        while v.is_running():
            step(10); v.sync(); show()
    if cv2 is not None: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()