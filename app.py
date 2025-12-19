from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict
import joblib
import numpy as np

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("ids_gb_model.pkl")

# -----------------------------
# FastAPI app config
# -----------------------------
app = FastAPI(
    title="Network Intrusion Detection System (IDS)",
    description="ML-based IDS using Gradient Boosting on the NSL-KDD dataset",
    version="1.0"
)

# -----------------------------
# Feature columns (TRAINING ORDER)
# -----------------------------
FEATURE_COLUMNS = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'protocol_type_icmp', 'protocol_type_tcp', 'protocol_type_udp',
    'service_IRC', 'service_X11', 'service_Z39_50', 'service_auth',
    'service_bgp', 'service_courier', 'service_csnet_ns', 'service_ctf',
    'service_daytime', 'service_discard', 'service_domain',
    'service_domain_u', 'service_echo', 'service_eco_i', 'service_ecr_i',
    'service_efs', 'service_exec', 'service_finger', 'service_ftp',
    'service_ftp_data', 'service_gopher', 'service_hostnames',
    'service_http', 'service_http_443', 'service_imap4',
    'service_iso_tsap', 'service_klogin', 'service_kshell',
    'service_ldap', 'service_link', 'service_login', 'service_mtp',
    'service_name', 'service_netbios_dgm', 'service_netbios_ns',
    'service_netbios_ssn', 'service_netstat', 'service_nnsp',
    'service_nntp', 'service_ntp_u', 'service_other',
    'service_pm_dump', 'service_pop_2', 'service_pop_3',
    'service_printer', 'service_private', 'service_remote_job',
    'service_rje', 'service_shell', 'service_smtp',
    'service_sql_net', 'service_ssh', 'service_sunrpc',
    'service_supdup', 'service_systat', 'service_telnet',
    'service_tftp_u', 'service_tim_i', 'service_time',
    'service_urp_i', 'service_uucp', 'service_uucp_path',
    'service_vmnet', 'service_whois',
    'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0',
    'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2',
    'flag_S3', 'flag_SF', 'flag_SH'
]

# -----------------------------
# Request schema
# -----------------------------
class NetworkTraffic(BaseModel):
    features: Dict[str, float] = Field(
        example={
            "duration": 0,
            "src_bytes": 181,
            "dst_bytes": 5450,
            "logged_in": 1,
            "count": 9,
            "srv_count": 9,
            "same_srv_rate": 1.0,
            "protocol_type_tcp": 1,
            "service_http": 1,
            "flag_SF": 1
        }
    )

# -----------------------------
# Utility: align input features
# -----------------------------
def align_features(input_features: Dict[str, float]) -> np.ndarray:
    """
    Align incoming feature dict to training feature order.
    Missing features -> 0
    """
    return np.array(
        [input_features.get(col, 0.0) for col in FEATURE_COLUMNS]
    ).reshape(1, -1)

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "num_features": len(FEATURE_COLUMNS)
    }

# -----------------------------
# Expose expected features
# -----------------------------
@app.get("/features")
def get_features():
    return {
        "num_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS
    }

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: NetworkTraffic):
    X = align_features(data.features)

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {
        "prediction": "attack" if prediction == 1 else "normal",
        "attack_probability": round(probability, 4)
    }