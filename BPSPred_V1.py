import re
import math
import joblib
import peptides
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem
from rasar import ra_similarity
from mordred import Calculator, descriptors
from rdkit.Chem import rdFingerprintGenerator
from SOCN_QSO import socn_qso_desc_calculator
from io import BytesIO
from PIL import Image

#inputs
inps = joblib.load("inputs.joblib")
inps1 = joblib.load("inputs1.joblib")
imgs = joblib.load("images.joblib")
img1 = Image.open(BytesIO(imgs["image1"]))
img2 = Image.open(BytesIO(imgs["image2"]))

#cytotoxicity
cyt_tr = inps[0]
cyt_te = inps[1]
ecfp_cyt = inps[7]
cyt_d = inps[9]
cytm = inps[12]
#cell death
cdh_tr = inps[2]
cdh_te = inps[3]
ecfp_cd = inps[8]
cd_d = inps[10]
cdm = inps[13]
#hemolysis
hem_tr = inps[4]
hem_te = inps[5]
ecfp_hem = inps1[0]
hem_d = inps[11]
hemm = inps[14]
#AAi values
desc_df = inps[6]

tr_cyto = cyt_tr
te_cyto = cyt_te

tr_hemo = hem_tr
te_hemo = hem_te

tr_cell = cdh_tr
te_cell = cdh_te

#similarity calculation for AD
def ecfp4_calculator(smiles_list):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    cal_fp = []
    for smi in smiles_list:
        mol = Chem.MolFromSequence(smi)
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            cal_fp.append(list(fp))
        else:
            cal_fp.append([None] * 4096)

    columns = [f"ECFP4_{i+1}" for i in range(4096)]
    cal_fp_df = pd.DataFrame(cal_fp, index=smiles_list, columns=columns)
    return cal_fp_df

def data_sort(frame, id):#id is the index of the new data frame
    df_val = pd.DataFrame(frame.apply(lambda row: [x[1] for x in sorted(zip(frame.columns, row), 
                                                                        key=lambda x: x[1], reverse=True)], 
                                                                        axis=1).tolist(), index=id)
    
    return df_val

def ad_analysis(df1_des, df2_des):
    print(df1_des.shape, df2_des.shape)
    sim = ra_similarity(df1_des, df2_des).similarity_calculation(method="Tanimoto Coefficient")
    sort_val = data_sort(sim, id=df2_des.index)
    sort_val1 = sort_val.iloc[:,0]
    AD_category = pd.Series(
            np.where(sort_val1 > 0.7, "Inside AD", "Outside AD"),
            index=sort_val1.index
        )
    return AD_category


#Descriptor calculation
def DDE_calculator(fastas, **kw):
    AA = kw['order'] if kw['order'] is not None else 'ACDEFGHIKLMNPQRSTVWY'
    myCodons = {
        'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2,
        'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
        'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6,
        'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
    }
    encodings = []
    diPeptides = ['DDE_' + aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['Sequence'] + diPeptides
    encodings.append(header)
    rawPairs = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    myTM = []
    for pair in rawPairs:
        myTM.append(
            (myCodons[pair[0]] / 61) *
            (myCodons[pair[1]] / 61)
        )

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for seq in fastas:
        sequence = re.sub('-', '', str(seq).upper())

        code = [sequence]
        tmpCode = [0] * 400

        for j in range(len(sequence) - 1):
            tmpCode[
                AADict[sequence[j]] * 20 +
                AADict[sequence[j + 1]]
            ] += 1

        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]

        myTV = []
        for j in range(len(myTM)):
            myTV.append(
                myTM[j] *
                (1 - myTM[j]) /
                (len(sequence) - 1)
            )

        for j in range(len(tmpCode)):
            tmpCode[j] = (
                tmpCode[j] - myTM[j]
            ) / math.sqrt(myTV[j])

        code = code + tmpCode
        encodings.append(code)
    return encodings

def calculate_descriptors(sequence, operator):
    amino_acids = list(sequence)
    amino_acids = [aa for aa in amino_acids if aa in desc_df.columns]
    if len(amino_acids) == 0:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float)
        )
    selected = desc_df[amino_acids]

    # Average
    if operator == "Avg":
        descriptor = selected.mean(axis=1)
    return descriptor

def aa_desc_calculator(sequences):
    results = []
    for seq in sequences:
        val = calculate_descriptors(seq, operator="Avg")
        row = val.to_dict()
        results.append(row)

    df = pd.DataFrame(results, index=sequences)
    df.columns = desc_df.iloc[:, 0].values.tolist()
    return df

def peptide_desc_calculator(sequences: list):
    all_desc = []
    for s in sequences:
        pep = peptides.Peptide(s)
        desc = pep.descriptors()
        all_desc.append(desc)
    df1 = pd.DataFrame(all_desc, index=sequences)
    return df1


def mordred_desc_cal(sequence: list):
    calc = Calculator(descriptors, ignore_3D=True)
    cal_des = []
    for i in sequence:
        mol = Chem.MolFromSequence(i)
        all_descriptors_dict = calc(mol).asdict()
        cal_des.append(all_descriptors_dict)
    mordred_desc_df = pd.DataFrame(cal_des, index=sequence)
    return mordred_desc_df


def ddor(sequences, residue):
    residue = residue.upper()
    if len(residue) != 1:
        raise ValueError("residue must be a single amino-acid letter")
    values = []
    for sequence in sequences:
        sequence = sequence.upper()
        positions = [
            pos for pos, char in enumerate(sequence)
            if char == residue
        ]
        if not positions:
            values.append(0)
            continue
        distances = []
        for i in range(len(positions) - 1):
            distances.append(
                positions[i + 1] - positions[i] - 1
            )
        distances.insert(0, positions[0])
        distances.append(
            len(sequence) - 1 - positions[-1]
        )
        denominator = sum(distances) + 1
        numerator = sum(d ** 2 for d in distances)
        ddor_value = numerator / denominator
        values.append(round(ddor_value, 2))
    return pd.DataFrame({
        f"DDR_{residue}": values
    }, index=sequences)

#Cytotoxic columns
cyt_des = ["ATSC3se", "SVGER5",	"OOBM850104",	"MEEJ800101",	"DDE_LN",	"DDE_LS",	"ROBB760107",	
           "CIC2",	"FilterItLogS",	"AATSC1m",	"AATS0s",	"WILM950104",	"ATSC1d",	"DDR_A",	"ProtFP5",	
           "SVGER8", "RACS820114",	"DDE_EW",	"DDE_FG",	"EState_VSA3",	"DDE_YA"]

#Cell-death columns
cdh_des = ["SlogP_VSA4",	"LEVM760103",	"DDE_EI",	"DDE_LT",	"ZIMJ680101",	"DDE_TL",	"EState_VSA2",
           "SOCNGM_lag1",	"ROBB760101",	"CHAM820102",	"NAKH900104",	"DDE_AR"]

#Hemolysis columns
hem_des = ["MEIH800103",	"FASG890101",	"PEOE_VSA6",	"Xc-3dv",	"PRIN1",	"NISK860101",	"CORJ870106",	"KRIW710101",	
           "NADH010104",	"CORJ870101",	"ATSC3v",	"KRIW790101",	"ATSC6dv",	"MEEJ810102",	"AATSC3v",	"OOBM850105",	"SlogP_VSA4",	
           "CHAM820102",	"Xpc-5dv",	"BROC820102",	"BIOV880102",	"ATSC6i",	"GUOD860101",	"PONP800107",	"ROBB790101",	"GRAR740102",	
           "PONP800102",	"ATSC8i",	"FUKS010104",	"HOPA770101",	"ATSC7Z",	"DDE_KG",	"PARJ860101",	"ATSC5Z",	
           "WILM950101",	"ATSC6pe",	"SOCNSW_lag1",	"ZHOH040103"]


def descriptor_calculator(sequences: list, type:str):
    dde = DDE_calculator(sequences, order='ACDEFGHIKLMNPQRSTVWY')
    dde_df = pd.DataFrame(dde[1:], columns=dde[0], index=sequences)

    aa_desc = aa_desc_calculator(sequences)
    pep_desc = peptide_desc_calculator(sequences)
    mordred_desc = mordred_desc_cal (sequences)
    ddr_a = ddor(sequences, residue='A')
    socn = socn_qso_desc_calculator(sequences)

    final_df = pd.concat([dde_df, aa_desc, pep_desc,  mordred_desc, ddr_a, socn], axis=1)
    if type == "cyt":
        df = final_df[cyt_des].copy()
    if type == "hem":
        df = final_df[hem_des].copy()
    if type == "cdh":
        df = final_df[cdh_des].copy()
    return df

def standerdization(df1, df2):
    avg = df1.mean()
    stdev = df1.std()
    std_df1 = (df1 - avg) / stdev
    std_df2 = (df2 - avg) / stdev
    return std_df1, std_df2


# ============================================================
# PEPTIDE INSIGHTS (descriptive only - NOT used in prediction)
# ============================================================
INS_NAVY, INS_TEAL, INS_PINK, INS_PURPLE, INS_GREY = "#062b55", "#009fc2", "#e84976", "#7954e8", "#9fb3c0"

# Kyte-Doolittle hydropathy scale
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

AA_CLASSES = {
    "Hydrophobic": "AVILMFW",
    "Polar": "STNQY",
    "Positive (+)": "KRH",
    "Negative (-)": "DE",
    "Special (G/P/C)": "GPC",
}
CLASS_COLORS = {
    "Hydrophobic": INS_TEAL,
    "Polar": INS_GREY,
    "Positive (+)": INS_PINK,
    "Negative (-)": INS_NAVY,
    "Special (G/P/C)": INS_PURPLE,
}
MAX_3D_LENGTH = 60  # keep interactive visualization responsive


def _composition_df(seq):
    rows = []
    for cls, letters in AA_CLASSES.items():
        for aa in letters:
            n = seq.count(aa)
            rows.append({"Residue": aa, "Class": cls, "Count": n,
                        "Percent": round(100 * n / len(seq), 2)})
    return pd.DataFrame(rows)


def _hydrophobicity_df(seq, window):
    df = pd.DataFrame({"Position": range(1, len(seq) + 1), "Residue": list(seq),
                      "KD": [KD_SCALE[a] for a in seq]})
    df["Smoothed"] = df["KD"].rolling(window, center=True, min_periods=1).mean()
    df["Type"] = np.where(df["KD"] >= 0, "Hydrophobic", "Hydrophilic")
    return df


def _charge_curve(seq):
    pep = peptides.Peptide(seq)
    ph = np.arange(0, 14.01, 0.5)
    return pd.DataFrame({"pH": ph, "Net charge": [pep.charge(pH=float(p)) for p in ph]})


@st.cache_data(show_spinner=False)
def _build_pdb(seq):
    """Generate an illustrative sequence-derived conformer, not a true fold prediction."""
    try:
        mol = Chem.MolFromSequence(seq)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) < 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1200)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
        return Chem.MolToPDBBlock(mol)
    except Exception:
        return None


def _viewer_html(pdb):
    """Minimal, large protein-cartoon viewer with mouse-drag rotation."""
    pdb_js = pdb.replace("\\", "\\\\").replace("`", "\\`")
    return f"""
    <div id="viewer" style="width:100%;height:760px;border:1px solid #e2eaf0;
         border-radius:16px;overflow:hidden;background:#ffffff;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
      const element = document.getElementById('viewer');
      const viewer = $3Dmol.createViewer(element, {{backgroundColor:'#ffffff', antialias:true}});
      viewer.addModel(`{pdb_js}`, 'pdb');
      viewer.setStyle({{}}, {{cartoon:{{color:'spectrum'}}}});
      viewer.zoomTo();
      viewer.zoom(1.35);
      viewer.render();
      viewer.spin('y', 0.45);
      window.addEventListener('resize', () => {{ viewer.resize(); viewer.render(); }});
    </script>
    """


def render_peptide_insights(sequence):
    seq = sequence.strip().upper()
    pep = peptides.Peptide(seq)
    net_charge = pep.charge(pH=7.4)
    gravy = float(np.mean([KD_SCALE[a] for a in seq]))
    pi = pep.isoelectric_point()
    mw = pep.molecular_weight()

    st.markdown("""\
        <div class="status-box">
            <div class="status-title">📊 Peptide Insights</div>
            <div class="status-text">Descriptive sequence properties shown for information only; they are not used in the safety predictions.</div>
        </div>
        """, unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Length", f"{len(seq)} aa")
    m2.metric("Molecular weight", f"{mw:,.1f} Da")
    m3.metric("Net charge (pH 7.4)", f"{net_charge:+.2f}")
    m4.metric("Isoelectric point", f"{pi:.2f}")
    m5.metric("GRAVY", f"{gravy:+.2f}")

    charge_label = "Cationic" if net_charge > 0.5 else "Anionic" if net_charge < -0.5 else "Near-neutral"
    hydro_label = "hydrophobic" if gravy > 0 else "hydrophilic"
    st.caption(f"Overall this peptide is **{charge_label}** and **{hydro_label}** (GRAVY is mean Kyte–Doolittle hydropathy).")
    tab1, tab2, tab3, tab4 = st.tabs(["🧬 Composition", "💧 Hydrophobicity", "⚡ Charge", "🧊 3D Structure"])

    with tab1:
        comp = _composition_df(seq)
        chart = alt.Chart(comp).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("Residue:N", sort=list("".join(AA_CLASSES.values())), axis=alt.Axis(labelAngle=0, title="Amino acid")),
            y=alt.Y("Percent:Q", title="Composition (%)"),
            color=alt.Color("Class:N", scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Residue", "Class", "Count", "Percent"],
        ).properties(height=340)
        st.altair_chart(chart, use_container_width=True)
        cls_counts = {c: sum(seq.count(a) for a in letters) for c, letters in AA_CLASSES.items()}
        cols = st.columns(len(cls_counts))
        for col, (c, n) in zip(cols, cls_counts.items()):
            col.metric(c, f"{100*n/len(seq):.1f}%", f"{n} residues", delta_color="off")

    with tab2:
        window = min(5, len(seq))
        hdf = _hydrophobicity_df(seq, window)
        base = alt.Chart(hdf)
        bars = base.mark_bar(opacity=0.85).encode(
            x=alt.X("Position:O", title="Residue position", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("KD:Q", title="Kyte–Doolittle hydropathy"),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["Hydrophobic", "Hydrophilic"], range=[INS_TEAL, INS_PINK]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Position", "Residue", "KD"],
        )
        line = base.mark_line(color=INS_NAVY, strokeWidth=2.5, point=True).encode(x="Position:O", y="Smoothed:Q", tooltip=["Position", alt.Tooltip("Smoothed:Q", format=".2f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#587284", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart((bars + line + zero).properties(height=360), use_container_width=True)
        st.caption(f"Bars: per-residue hydropathy. Line: centered {window}-residue moving average. Positive values are hydrophobic.")

    with tab3:
        n_pos = sum(seq.count(a) for a in "KR")
        n_his = seq.count("H")
        n_neg = sum(seq.count(a) for a in "DE")
        c1, c2, c3 = st.columns(3)
        c1.metric("Lysine + Arginine residues", n_pos)
        c2.metric("Aspartate + Glutamate residues", n_neg)
        c3.metric("Histidine residues", n_his)
        cdf = _charge_curve(seq)
        curve = alt.Chart(cdf).mark_line(color=INS_TEAL, strokeWidth=3).encode(x=alt.X("pH:Q", scale=alt.Scale(domain=[0, 14])), y=alt.Y("Net charge:Q"), tooltip=["pH", alt.Tooltip("Net charge:Q", format=".2f")])
        marker = alt.Chart(pd.DataFrame({"pH": [7.4], "Net charge": [net_charge]})).mark_point(color=INS_PINK, size=140, filled=True).encode(x="pH:Q", y="Net charge:Q")
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#587284", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart((curve + marker + zero).properties(height=340), use_container_width=True)
        st.caption("Net charge across pH. Pink marker shows pH 7.4. Charge is estimated using the peptide library's pKa assumptions.")

    with tab4:
        if len(seq) > MAX_3D_LENGTH:
            st.info(f"The interactive 3D model is limited to {MAX_3D_LENGTH} residues for responsive rendering (submitted: {len(seq)}).")
        else:
            with st.spinner("Building 3D molecular model…"):
                pdb = _build_pdb(seq)
            if not pdb:
                st.warning("A 3D model could not be generated for this sequence.")
            else:
                components.html(_viewer_html(pdb), height=780, scrolling=False)


# CUSTOM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #ffffff;
}
.home-title {
    font-size: 50px;
    font-weight: 850;
    letter-spacing: -1.4px;
    margin: 0;
    color: #000;

    animation: slideFromRight 1.2s ease-out forwards;
}

.home-title .bps {
    color: #062b55;
}

.home-title .pred {
    color: #009fc2;
}

@keyframes slideFromRight {
    0% {
        opacity: 0;
        transform: translateX(120px);
    }

    100% {
        opacity: 1;
        transform: translateX(0);
    }
}

/* ============================================================
   MAIN CONTAINER
   ============================================================ */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* ============================================================
   Navigation
   ============================================================ */
section[data-testid="stSidebar"] .stRadio > label {
display: none;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 5px;
    width: 100%;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    width: 100%;
    box-sizing: border-box;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 0;
    color: #EEF5FA;
    font-size: 14px;
    font-weight: 600;
}

/* ============================================================
   Hover
   ============================================================ */

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: transparent;
    border-color: transparent;
    color: #07345f !important;
}


/* ============================================================
   Selected page
   ============================================================ */

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: #ffffff;
    border: 1px solid #d2e8ef;
    color: #07345f !important;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(7,62,91,0.06);
}

/* ============================================================
   SIDEBAR INFORMATION
   ============================================================ */
.sidebar-info {
    background: #f5fcff;
    border: 1px solid #cdebf4;
    border-radius: 13px;
    padding: 14px;
    margin-top: 15px;
    color: #587284;
    font-size: 11px;
    line-height: 1.55;
}
.sidebar-info strong {
    color: #009fc2;
}

/* ============================================================
   TOP HOME HEADER
   ============================================================ */
.home-header {
    display: flex;
    align-items: center;
    gap: 22px;
    padding: 8px 0 20px 0;
}
.home-logo {
    width: 105px;
    height: 105px;
    object-fit: contain;
}
.home-title {
    font-size: 50px;
    font-weight: 850;
    letter-spacing: -1.4px;
    margin: 0;
    color: #000000;
}
.home-title .bps {
    color: #062b55;
}
.home-title .pred {
    color: #009fc2;
}
.home-subtitle {
    color: #60798b;
    font-size: 20px;
    line-height: 2;
    max-width: 930px;
    margin-top: 5px;
}

/* ============================================================
   HERO
   ============================================================ */
.hero {
    border-radius: 22px;
    padding: 30px 34px;
    background:
        linear-gradient(
            110deg,
            #061a35 0%,
            #083b6c 55%,
            #008fa7 100%
        );
    box-shadow:
        0 12px 35px rgba(3, 40, 75, 0.17);
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
}
.hero:after {
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    border-radius: 50%;
    right: -80px;
    top: -120px;
    border: 1px solid rgba(66,235,255,0.25);
    box-shadow:
        0 0 0 30px rgba(66,235,255,0.03),
        0 0 0 60px rgba(66,235,255,0.025);
}
.hero-title {
    color: white;
    font-size: 29px;
    font-weight: 800;
    margin-bottom: 8px;
}
.hero-title span {
    color: #45eaff;
}
.hero-text {
    color: #d7f5fb;
    font-size: 14px;
    line-height: 1.65;
    max-width: 900px;
}

/* ============================================================
   SECTION TITLES
   ============================================================ */
.section-title {
    font-size: 35px;
    font-weight: 800;
    color: #07345f;
    margin-bottom: 5px;
}
.section-subtitle {
    color: #71889a;
    font-size: 20px;
    margin-bottom: 20px;
}

/* ============================================================
   CARDS
   ============================================================ */
.info-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid #d7eaf1;
    border-radius: 18px;
    padding: 22px;
    min-height: 175px;
    box-shadow:
        0 8px 25px rgba(9,67,98,0.065);
    transition: 0.2s;
}
.info-card:hover {
    transform: translateY(-2px);
    border-color: #45cfe8;
    box-shadow:
        0 12px 32px rgba(0,165,200,0.13);
}
.endpoint-icon {
    font-size: 42px;
    margin-bottom: 10px;
}
.card-title {
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 8px;
}
.card-text {
    text-align: justify;
    font-size: 18px;
    font-weight: 600;
    line-height: 1.5;
}

/* ============================================================
   Endpoint Accents
   ============================================================ */
.cyto-card {
    border-top: 5px solid #e84976;
}
.hemo-card {
    border-top: 5px solid #00a99d;
}
.death-card {
    border-top: 5px solid #7954e8;
}

/* ============================================================
   PREDICTION WORKSPACE
   ============================================================ */
.workspace {
    background: white;
    border: 1px solid #d4e8f0;
    border-radius: 21px;
    padding: 27px;
    box-shadow:
        0 10px 30px rgba(7,62,91,0.07);
    margin-top: 25px;
}

/* ============================================================
   MAIN MODE SELECTOR
   ============================================================ */
div[data-testid="stRadio"] > div {
    gap: 14px;
    width: 100%;
}

div[data-testid="stRadio"] label {
    font-size: 19px !important;
    font-weight: 700 !important;
}
.mode-description {
    background: #f0faff;
    border: 1px solid #cdebf4;
    border-radius: 12px;
    padding: 12px 16px;
    color: #587284;
    font-size: 16px;
    margin-bottom: 20px;
}

/* ============================================================
   BUTTONS
   ============================================================ */
div.stButton > button {
    border-radius: 11px;
    border: 1px solid #079fbd;
    background:
        linear-gradient(
            100deg,
            #056ac8,
            #00a9c7
        );
    color: white;
    font-weight: 750;
    min-height: 45px;
    transition: 0.2s;
}
div.stButton > button:hover {
    border-color: #00d5ed;
    box-shadow:
        0 7px 20px rgba(0,165,210,0.25);
    transform: translateY(-1px);
}
/* ============================================================
   DOWNLOAD BUTTONS
   ============================================================ */

div.stDownloadButton > button {
    border-radius: 11px !important;
    border: 1px solid #079fbd !important;
    background: linear-gradient(
        100deg,
        #056ac8,
        #00a9c7
    ) !important;
    color: white !important;
    font-weight: 750 !important;
    min-height: 45px !important;
    transition: 0.2s !important;
}

div.stDownloadButton > button:hover {
    border-color: #00d5ed !important;
    box-shadow: 0 7px 20px rgba(0,165,210,0.25) !important;
    transform: translateY(-1px) !important;
    color: white !important;
}

div.stDownloadButton > button:focus {
    background: linear-gradient(
        100deg,
        #056ac8,
        #00a9c7
    ) !important;
    color: white !important;
    border-color: #079fbd !important;
}

/* ============================================================
   TEXT AREA
   ============================================================ */
textarea {
    border-radius: 13px !important;
    border: 1px solid #cfe4ed !important;
}
textarea:focus {
    border-color: #00aaca !important;
    box-shadow:
        0 0 0 2px rgba(0,180,220,0.10) !important;
}

/* ============================================================
   RESULT CARDS
   ============================================================ */
.result-card {
    border-radius: 17px;
    padding: 21px;
    background: white;
    min-height: 150px;
    border: 1px solid #d9e9ef;
    box-shadow:
        0 7px 22px rgba(0,50,80,0.07);
}
.result-heading {
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 1.1px;
    color: #6b8190;
}
.result-value {
    font-size: 28px;
    font-weight: 800;
    margin-top: 10px;
}
.safe {
    color: #008f76;
}
.unsafe {
    color: #d93662;
}
.warning {
    color: #d18b00;
}

/* ============================================================
   STATUS
   ============================================================ */
.status-box {
    border-radius: 17px;
    padding: 19px 23px;
    background:
        linear-gradient(
            100deg,
            #e9fbff,
            #f5fdff
        );
    border-left: 6px solid #00aeca;
    margin: 20px 0;
}
.status-title {
    font-size: 18px;
    font-weight: 800;
    color: #07345f;
}
.status-text {
    color: #587284;
    margin-top: 5px;
    font-size: 13px;
    line-height: 1.6;
}

/* ============================================================
   FEATURE STRIP
   ============================================================ */
.feature-strip {
    display: flex;
    justify-content: space-between;
    align-items: stretch;
    gap: 18px;
    margin-top: 28px;
    padding: 20px 10px;
}
.feature-item {
    flex: 1;
    text-align: center;
    font-size: 17px;
    font-weight: 500;
    line-height: 1.5;
    padding: 18px 12px;
}
.feature-item:first-line {
    font-size: 32px;
}
.feature-item strong {
    font-size: 19px;
    font-weight: 800;
    color: #062b55;
}
/* ============================================================
   METRICS
   ============================================================ */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #d8ebf2;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 6px 20px rgba(7,62,91,0.045);
}
div[data-testid="stMetricLabel"] {
    font-size: 16px !important;
}
div[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricDelta"] {
    font-size: 15px !important;
}

/* ============================================================
   DATAFRAME
   ============================================================ */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
}

/* ============================================================
   Quick likks
   ============================================================ */
.sidebar-section-title {
    font-size: 16px;
    font-weight: 800;
    color: #062b55;
    letter-spacing: 0.8px;
    margin: 8px 0 12px 4px;
    padding-bottom: 8px;
}
/* ============================================================
   DIVIDER
   ============================================================ */
hr {
    border-color: #dcecf2;
}
/* Remove divider inside sidebar */
section[data-testid="stSidebar"] hr {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)




# PAGE CONFIGURATION
st.set_page_config(
    page_title="BPSPred | Bioactive Peptide Safety Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR
with st.sidebar:

    # SIDEBAR LOGO
    st.markdown(
        '<div class="sidebar-logo">',
        unsafe_allow_html=True
    )

    
    st.image(
        img1,
        width=250
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # NAVIGATION
    st.sidebar.markdown(
    """
    <div class="sidebar-section-title">
        Quick Links
    </div>
    """,
    unsafe_allow_html=True
    )
    
    page = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "🧠  About the Models",
            "📊  Training and Test Sets",
            "📖  User Manual",
            "📧  Contact Us"
        ],
        label_visibility="collapsed"
    )

    # SIDEBAR BANNER
    st.markdown(
        '<div class="sidebar-banner">',
        unsafe_allow_html=True
    )

    
    st.image(
        img2,
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # VERSION
    st.markdown(
        '<div style="text-align: center; '
        'margin-top: 8px; '
        'font-size: 16px; '
        'color: #777; '
        'font-weight: 700;">'
        'Version_1.0'
        '</div>',
        unsafe_allow_html=True
    )


# HOME
if page == "🏠  Home":

    st.markdown(
        """
        <div class="home-title">
            Welcome to <span class="bps">BPS</span><span class="pred">Pred</span><span style="color: black;">_v1</span>
        </div>

        <div class="home-subtitle">
            BPSPred is a multi-endpoint prediction platform based on machine learning, 
            design to evaluate the safety profile of bioactive peptides. It predicts 
            cytotoxicity, hemolysis and cell death potential, helping researchers 
            screen safer peptides for therapeutic and nutraceutical applications.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.space("small")

    # ENDPOINT CARDS
    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="info-card cyto-card">

            <div class="endpoint-icon">
                    🧬
            </div>

            <div class="card-title">
                    Cytotoxicity
            </div>

            <div class="card-text">
                    Predict cytotoxiity toward cell lines
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            """
            <div class="info-card hemo-card">

            <div class="endpoint-icon">
                    🩸
            </div>

            <div class="card-title">
                    Hemolysis
            </div>

            <div class="card-text">
                    Predict red blood cell lysis potential
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            """
            <div class="info-card death-card">

            <div class="endpoint-icon">
                    🧫
            </div>

            <div class="card-title">
                    Cell death
            </div>

            <div class="card-text">
                    Predict cell death initiation potential
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )



    #Prediction workspace
    st.space("small")
    st.markdown(
        """
        <div class="section-title">
            🔬 Peptide Safety Prediction
        </div>

        <div class="section-subtitle">
            Choose a prediction mode and submit your peptide sequence
            or Excel dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    # PREDICTION MODE
    prediction_mode = st.radio(
        "Prediction mode",
        [
            "🧬  Single Peptide Prediction",
            "📑  Batch Prediction"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    # SINGLE PEPTIDE PREDICTION
    if prediction_mode == "🧬  Single Peptide Prediction":

        st.markdown(
            """
            <div class="mode-description">
                Enter one peptide sequence using 20 standard
                one-letter natural amino acid codes only.
            </div>
            """,
            unsafe_allow_html=True
        )
        def reset_sequence():
            st.session_state["sequence_input"] = ""

        sequence = st.text_area(
            "Peptide sequence",
            placeholder="Example: KLLKLLLKLLK",
            height=80,
            label_visibility="collapsed",
            key="sequence_input"
        )

    
        # SEQUENCE VALIDATION
        clean_sequence = ""
        if sequence:
            clean_sequence = sequence
            valid_aa = set(
                "ACDEFGHIKLMNPQRSTVWY"
            )
            invalid = [
                aa
                for aa in clean_sequence
                if aa not in valid_aa
            ]
            if invalid:
                st.error(
                    "Invalid Sequence"

                )
            elif len(clean_sequence) < 2:
                st.error(
                    "Single amino acid is not accepted"
                )
            else:
                st.success(
                    f"Sequence accepted  •  "
                    f"Length: {len(clean_sequence)} residues"
                )

        # BUTTONS
        col_a, col_b = st.columns([3, 1])

        with col_a:
            predict = st.button(
                "▶  Predict Safety Profile",
                use_container_width=True,
                key="single_predict"
            )

        with col_b:
            reset = st.button(
                "↻  Reset",
                use_container_width=True,
                key="single_reset",
                on_click=reset_sequence
            )


        # DEMONSTRATION RESULT
        if predict and clean_sequence:

            valid_aa = set(
                "ACDEFGHIKLMNPQRSTVWY"
            )

            if all(
                aa in valid_aa
                for aa in clean_sequence
            ):
                ecfp_te = ecfp4_calculator([sequence])
                #cyt
                cyt_df = descriptor_calculator([clean_sequence], type="cyt")
                print(cyt_df.columns)
                print(cyt_d.iloc[:,:-1].columns)
                __, cyt_std_test = standerdization(cyt_d.iloc[:,:-1], cyt_df)
                pred_cyt = cytm.predict(cyt_std_test)

                if pred_cyt[0] == 0:
                    cyt_status = "Less Cytotoxic"
                    cyt_class = "safe"
                else:
                    cyt_status = "More Cytotoxic"
                    cyt_class = "unsafe"

                ad_stat_cyt= ad_analysis(ecfp_cyt, ecfp_te)

                #hem
                hem_df = descriptor_calculator([clean_sequence], type="hem")
                __, hem_std_test = standerdization(hem_d.iloc[:,:-1], hem_df)
                pred_hem = hemm.predict(hem_std_test)

                if pred_hem[0] == 0:
                    hem_status = "Less Hemolytic"
                    hem_class = "safe"
                else:
                    hem_status = "More Hemolytic"
                    hem_class = "unsafe"

                ad_stat_hem= ad_analysis(ecfp_hem, ecfp_te)

                #cdh
                cdh_df = descriptor_calculator([clean_sequence], type="cdh")
                __, cdh_std_test = standerdization(cd_d.iloc[:,:-1], cdh_df)
                pred_cdh = cdm.predict(cdh_std_test)

                if pred_cdh[0] == 0:
                    cdh_status = "Less Cell death Initiating"
                    cdh_class = "safe"
                else:
                    cdh_status = "More Cell death Initiating"
                    cdh_class = "unsafe"

                ad_stat_cdh= ad_analysis(ecfp_cd, ecfp_te)

                # SAFETY PROFILE HEADER
                st.markdown(
                    """
                    <div class="status-box">

                    <div class="status-title">
                        🛡️ BPSPred Safety Profile
                    </div>

                    <div class="status-text">
                        Prediction generated for the submitted
                        peptide sequence.
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # RESULT CARDS
                r1, r2, r3 = st.columns(3)

                with r1:

                    st.markdown(f"""
                        <div class="result-card">
                        <div class="result-heading">
                        CYTOTOXICITY
                        </div>
                        <div class="result-value {cyt_class}">
                        {cyt_status}
                        </div>
                        <div style="
                        color:#728997;
                        font-size:16px;
                        margin-top:10px;
                        ">AD Status: 
                        {ad_stat_cyt.iloc[0]}
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with r2:

                    st.markdown(f"""
                        <div class="result-card">
                        <div class="result-heading">
                        HEMOLYSIS
                        </div>
                        <div class="result-value {hem_class}">
                        {hem_status}
                        </div>
                        <div style="
                        color:#728997;
                        font-size:16px;
                        margin-top:10px;
                        ">AD Status: 
                        {ad_stat_hem.iloc[0]}
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with r3:

                    st.markdown(f"""
                        <div class="result-card">
                        <div class="result-heading">
                        CELL DEATH
                        </div>
                        <div class="result-value {cdh_class}">
                        {cdh_status}
                        </div>
                        <div style="
                        color:#728997;
                        font-size:16px;
                        margin-top:10px;
                        "> AD Status: 
                        {ad_stat_cdh.iloc[0]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ---------------------------------------------------
                # NEW: PEPTIDE INSIGHTS (composition, hydrophobicity,
                # charge, 3D structure) - not used in prediction
                # ---------------------------------------------------
                render_peptide_insights(clean_sequence)

    # BATCH PREDICTION
    else:

        st.markdown(
            """
            <div class="mode-description">
                Upload an Excel file containing a
                <b>Sequence</b> column to predict multiple
                peptides simultaneously.
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload Excel dataset",
            type=["xlsx"],
            help="The Excel file must contain a column named Sequence.",
            key="batch_upload"
        )

        if uploaded_file:

            try:

                df = pd.read_excel(
                    uploaded_file
                )


                if "Sequence" not in df.columns:

                    st.error(
                        "The uploaded Excel file must contain "
                        "a column named 'Sequence'."
                    )

                else:

                    st.success(
                        f"Dataset loaded successfully  •  "
                        f"{len(df)} peptide(s) detected"
                    )

                    st.markdown(
                        '<div class="section-subtitle">'
                        'Preview of uploaded dataset'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.dataframe(
                        df.head(10),
                        use_container_width=True,
                        hide_index=True
                    )

                    run_batch = st.button(
                        "▶  Run Batch Prediction",
                        use_container_width=True,
                        key="run_batch"
                    )

                    if run_batch:

                        result_df = df.copy()
                        sequences = df["Sequence"].tolist()

                        results = []
                        for sequence in sequences:

                            clean_sequence = sequence.strip()
                            ecfp_te = ecfp4_calculator([clean_sequence])

                            cyt_df = descriptor_calculator([clean_sequence], type="cyt")
                            _, cyt_std_test = standerdization(
                                cyt_d.iloc[:, :-1],
                                cyt_df
                            )

                            pred_cyt = cytm.predict(cyt_std_test)

                            if pred_cyt[0] == 0:
                                cyt_status = "Less Cytotoxic"
                            else:
                                cyt_status = "More Cytotoxic"

                            ad_stat_cyt = ad_analysis(ecfp_cyt, ecfp_te)

                            hem_df = descriptor_calculator([clean_sequence], type="hem")
                            _, hem_std_test = standerdization(
                                hem_d.iloc[:, :-1],
                                hem_df
                            )

                            pred_hem = hemm.predict(hem_std_test)

                            if pred_hem[0] == 0:
                                hem_status = "Less Hemolytic"
                            else:
                                hem_status = "More Hemolytic"

                            ad_stat_hem = ad_analysis(ecfp_hem, ecfp_te)

                            cdh_df = descriptor_calculator([clean_sequence], type="cdh")
                            _, cdh_std_test = standerdization(
                                cd_d.iloc[:, :-1],
                                cdh_df
                            )

                            pred_cdh = cdm.predict(cdh_std_test)

                            if pred_cdh[0] == 0:
                                cdh_status = "Less Cell death Initiating"
                            else:
                                cdh_status = "More Cell death Initiating"

                            ad_stat_cdh = ad_analysis(ecfp_cd, ecfp_te)


                            results.append({
                                "Sequence": clean_sequence,

                                "Cytotoxicity": cyt_status,
                                "Cytotoxicity_AD": ad_stat_cyt.values[0],

                                "Hemolysis": hem_status,
                                "Hemolysis_AD": ad_stat_hem.values[0],

                                "Cell death": cdh_status,
                                "Cell death_AD": ad_stat_cdh.values[0]
                            })

                        batch_result_df = pd.DataFrame(results)


                        # Batch prediction
                        result_df = batch_result_df

                        st.success(
                            "Batch prediction completed successfully."
                        )

                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        csv_data = result_df.to_csv(
                            index=False
                        )

                        st.download_button(
                            "Download Prediction Results",
                            data=csv_data,
                            file_name="BPSPred_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_results"
                        )

            except Exception as e:
                st.error(
                    f"Unable to read the uploaded Excel file: {e}"
                )



# ABOUT THE MODEL
elif page == "🧠  About the Models":

    st.markdown(
        """
        <div class="home-title">
            About the <span> Models</span>
        </div>

        <div class="home-subtitle">
            Machine learning-assisted multi-endpoint framework
            for bioactive peptide safety assessment.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="info-card cyto-card">


            <div class="card-title">
                Cytotoxicity Model
            </div>

            <div class="card-text">
                <p>
                The BPSPred cytotoxic model was developed using curated peptide datasets collected from multiple
                established databases, including DBAASP v3 and DCTpep, obtaining 236 peptides, comprising 152 cytotoxic 
                and 84 non-cytotoxic peptides. Peptides with a 50% cytotoxicity concentration of ≤100 μM were classified 
                as more cytotoxic, while those above 100 μM were considered less cytotoxic. 
                </p>
                <hr>
                <p>
                The model development incorporated combined molecular representations, including AAIindices, Mordred, and integrated 
                peptide descriptors with an accuracy of 89%.
                </p>
                <hr>
                <p>
                The best-performing model was selected based on MCC robustness, providing a reliable computational
                approach for assessing the cytotoxic potential of bioactive peptides.
                </p>
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="info-card hemo-card">


            <div class="card-title">
                Hemolysis Model
            </div>

            <div class="card-text">
                <p>
                The BPSPred hemolysis model was developed using curated peptide datasets collected from multiple established databases, 
                including DBAASP v3 and Hemolytik 2.0, obtaining 2,384 peptides, comprising 1,088 hemolytic and 1,296 
                non-hemolytic peptides. Peptides with a 50% hemolysis concentration of ≤100 μM were classified as more hemolytic, 
                while those above 100 μM were considered less hemolytic.
                </p>
                <hr>
                <p>
                The model development incorporated combined molecular representations, including AAIndices, Mordred, and integrated 
                peptide descriptors with an accuracy of 85%.
                </p>
                <hr>
                <p>
                The best-performing model was selected based on MCC robustness, providing a reliable computational 
                approach for assessing the hemolytic potential of bioactive peptides.
                </p>
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="info-card death-card">


            <div class="card-title">
                Cell death Model
            </div>

            <div class="card-text">
                <p>
                The BPSPred cell death model was developed using peptides collected from curated DBAASP v3 and DCTpep databases, 
                obtaining 344 peptides, comprising 198 more cell death initiating
                and 146 less cell death initating peptides. Peptides with a 50% cell death concentration of ≤100 μM were classified 
                as more cell death initating, while above 100 μM were considered less cell death initiating.
                </p>
                <hr>
                <p>
                The model development incorporated combined molecular representations, including AAIndices, RDKit, 
                and integrated peptide descriptors with an accuracy of 84%.
                </p>
                <hr>
                <p>
                The best-performing model was selected based on MCC robustness, 
                providing a reliable computational approach for assessing the cell death potential of bioactive peptides.
                </p>
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # FEATURE STRIP
    st.space("small")
    st.markdown(
        """
        <div class="feature-strip">

        <div class="feature-item">
            🧬<br>
            <strong>Multi-Endpoint</strong><br>
            Three safety endpoints
        </div>

        <div class="feature-item">
            🧠<br>
            <strong>Machine Learning</strong><br>
            Data-driven prediction
        </div>

        <div class="feature-item">
            ⚡<br>
            <strong>Rapid</strong><br>
            Fast computation
        </div>

        <div class="feature-item">
            📑<br>
            <strong>Batch Analysis</strong><br>
            Excel-based prediction
        </div>

        <div class="feature-item">
            🛡️<br>
            <strong>Safety Profiling</strong><br>
            Integrated assessment
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )



# Training and Test Sets
elif page == "📊  Training and Test Sets":

    st.markdown(
        """
        <div class="home-title">
            Training and <span>Test Sets</span>
        </div>

        <div class="home-subtitle">
            Datasets used for model training
            and independent validation of the BPSPred
            safety endpoints.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # CYTOTOXICITY
    st.markdown("### 50% Cytotoxicity")

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "👁 View Training Dataset",
            key="view_train_cyto",
            use_container_width=True
        ):

            st.dataframe(
                tr_cyto,
                use_container_width=True,
                height=400
            )

    with c2:

        csv_train_cyto = tr_cyto.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Training Dataset",
            data=csv_train_cyto,
            file_name="cytotoxicity_training_dataset.csv",
            mime="text/csv",
            key="download_train_cyto",
            use_container_width=True
        )

    c3, c4 = st.columns(2)

    with c3:

        if st.button(
            "👁 View Test Dataset",
            key="view_test_cyto",
            use_container_width=True
        ):

            st.dataframe(
                te_cyto,
                use_container_width=True,
                height=400
            )

    with c4:

        csv_test_cyto = te_cyto.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Test Dataset",
            data=csv_test_cyto,
            file_name="cytotoxicity_test_dataset.csv",
            mime="text/csv",
            key="download_test_cyto",
            use_container_width=True
        )

    st.divider()

    # HEMOLYSIS
    st.markdown("### 50% Hemolysis")

    c5, c6 = st.columns(2)

    with c5:

        if st.button(
            "👁 View Training Dataset",
            key="view_train_hemo",
            use_container_width=True
        ):

            st.dataframe(
                tr_hemo,
                use_container_width=True,
                height=400
            )

    with c6:

        csv_train_hemo = tr_hemo.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Training Dataset",
            data=csv_train_hemo,
            file_name="hemolytic_training_dataset.csv",
            mime="text/csv",
            key="download_train_hemo",
            use_container_width=True
        )

    c7, c8 = st.columns(2)

    with c7:

        if st.button(
            "👁 View Test Dataset",
            key="view_test_hemo",
            use_container_width=True
        ):

            st.dataframe(
                te_hemo,
                use_container_width=True,
                height=400
            )

    with c8:

        csv_test_hemo = te_hemo.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Test Dataset",
            data=csv_test_hemo,
            file_name="hemolytic_test_dataset.csv",
            mime="text/csv",
            key="download_test_hemo",
            use_container_width=True
        )

    st.divider()

    # CELL-DEATH
    st.markdown("### 50% Cell death")

    c9, c10 = st.columns(2)

    with c9:

        if st.button(
            "👁 View Training Dataset",
            key="view_train_cell",
            use_container_width=True
        ):

            st.dataframe(
                tr_cell,
                use_container_width=True,
                height=400
            )

    with c10:

        csv_train_cell = tr_cell.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Training Dataset",
            data=csv_train_cell,
            file_name="cell_death_training_dataset.csv",
            mime="text/csv",
            key="download_train_cell",
            use_container_width=True
        )

    c11, c12 = st.columns(2)

    with c11:

        if st.button(
            "👁 View Test Dataset",
            key="view_test_cell",
            use_container_width=True
        ):

            st.dataframe(
                te_cell,
                use_container_width=True,
                height=400
            )

    with c12:

        csv_test_cell = te_cell.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Test Dataset",
            data=csv_test_cell,
            file_name="cell_death_test_dataset.csv",
            mime="text/csv",
            key="download_test_cell",
            use_container_width=True
        )



# User Manual
elif page == "📖  User Manual":

    st.markdown(
        """
        <div class="home-title">
            User <span>Manual</span>
        </div>

        <div class="home-subtitle">
            Guidelines for using BPSPred for multi-endpoint
            safety assessment of bioactive peptides.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <h3 style="color:#0F4C81;">1. Single Peptide Prediction</h3>

    <p>
    Navigate to the prediction section and select
    <strong>Single Peptide Prediction</strong>. Enter a valid peptide
    sequence using the standard single-letter amino acid codes
    (<strong>A–Z</strong> with the 20 standard amino acids) in the
    input box. After entering the sequence, click the
    <strong>Predict Safety Profile</strong> button to initiate the analysis.
    BPSPred evaluates the submitted peptide using three independent
    safety endpoints: <strong>Cytotoxicity</strong>,
    <strong>Hemolytic Activity</strong>, and
    <strong>Cell death</strong>. The corresponding predicted safety
    status and Applicability Domain (AD) information are displayed
    for each endpoint.
    </p>

    <hr>

    <h3 style="color:#0F4C81;">2. Batch Prediction</h3>

    <p>
    To analyze multiple peptide sequences simultaneously, select
    <strong>Batch Prediction</strong> and upload an
    <strong>Excel (.xlsx) file</strong> containing the <strong>Sequence</strong> column
    in the required input format.
    After uploading the file, click <strong>Predict Safety Profile</strong> to
    process all submitted peptide sequences. The prediction results
    for the three safety endpoints are displayed in a tabular format.
    The resulting dataset can also be <strong>downloaded</strong>
    for further analysis and documentation.
    </p>

    <hr>

    <h3 style="color:#0F4C81;">3. Interpreting the Results</h3>

    <p>
    BPSPred provides predictions for three important safety-related
    endpoints of bioactive peptides.
    The <strong>Cytotoxicity Status</strong> indicates whether the
    peptide is predicted to be <strong>50% Cytotoxic</strong> or
    <strong> Less cytotoxic</strong>.
    The <strong>Hemolysis Status</strong> indicates whether the
    peptide is predicted to exhibit <strong>50% Hemolytic</strong> or
    <strong>Less hemolytic</strong> activity.
    The <strong>Cell death Status</strong> indicates whether the
    peptide is predicted to have <strong> 50% Cell death inducing</strong>
    or <strong>Less cell eath inducing</strong> activity.
    </p>

    <p>
    The <strong>Applicability Domain (AD) Status</strong> indicates
    whether the query peptide falls within the chemical and
    structural space represented by the peptides used
    for model development. Predictions within the AD are considered
    more reliable, whereas <strong>Outside AD</strong> predictions
    should be interpreted with caution.
    </p>

    <p>
    The predictions generated by BPSPred are intended to support
    preliminary safety assessment and prioritization. Users should
    consider the model predictions together with experimental
    evidence, physicochemical properties, and other relevant
    biological information before drawing final conclusions.
    </p>

    <hr>

    <h3 style="color:#0F4C81;">4. Terms of Use</h3>

    <p>
    <strong>BPSPred</strong> has been developed
    <strong>solely for research and academic purposes</strong>.
    The predictions generated by this server should not be considered
    a substitute for experimental validation, clinical assessment,
    or regulatory evaluation.For technical issues, calculation-related
    problems, or further information regarding the server, please
    contact the individuals listed in the <strong>Contact Us</strong>
    section.
    </p>

    <hr>
    
    <h3 style="color:#0F4C81;">5. Cite Us</h3>

    <p>
    <strong>Reference</strong> will be available soon.
    </p>

    """, unsafe_allow_html=True)



# CONTACT US
elif page == "📧  Contact Us":
    st.markdown(
        """
        <div class="home-title">
            Contact <span>Us</span>
        </div>

        <div class="home-subtitle">
            For queries, technical assistance,
            or information regarding BPSPred Contact individuals.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <h3 style="color:#0F4C81;">Contact information will be updated soon!</h3>
    <div style="line-height:1.7; margin-left:10px; margin-bottom:25px;">
    <span style="font-size:22px; font-weight:bold;">
    </span><br>
    </a>
    </div>
    """, unsafe_allow_html=True)
