"""
NSL-KDD Data Loader for NeuroIDS-Sat v2

Optimized with float32 for faster processing.

Dataset: https://www.unb.ca/cic/datasets/nsl.html

Author: Toby R. Davis
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict


# Column names and mappings
ATTACK_CATEGORIES = {
    'normal': 0,
    # DoS
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
    'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1,
    # Probe
    'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2, 'saint': 2,
    # R2L
    'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3, 'phf': 3,
    'spy': 3, 'warezclient': 3, 'warezmaster': 3, 'sendmail': 3, 'named': 3,
    'snmpgetattack': 3, 'snmpguess': 3, 'xlock': 3, 'xsnoop': 3, 'worm': 3,
    # U2R
    'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4, 'httptunnel': 4,
    'ps': 4, 'sqlattack': 4, 'xterm': 4
}

PROTOCOLS = {'tcp': 0, 'udp': 1, 'icmp': 2}

SERVICES = {
    'http': 0, 'smtp': 1, 'finger': 2, 'domain_u': 3, 'auth': 4, 'telnet': 5,
    'ftp': 6, 'eco_i': 7, 'ntp_u': 8, 'ecr_i': 9, 'other': 10, 'private': 11,
    'pop_3': 12, 'ftp_data': 13, 'rje': 14, 'time': 15, 'mtp': 16, 'link': 17,
    'remote_job': 18, 'gopher': 19, 'ssh': 20, 'name': 21, 'whois': 22,
    'domain': 23, 'login': 24, 'imap4': 25, 'daytime': 26, 'ctf': 27, 'nntp': 28,
    'shell': 29, 'IRC': 30, 'nnsp': 31, 'http_443': 32, 'exec': 33, 'printer': 34,
    'efs': 35, 'courier': 36, 'uucp': 37, 'klogin': 38, 'kshell': 39, 'echo': 40,
    'discard': 41, 'systat': 42, 'supdup': 43, 'iso_tsap': 44, 'hostnames': 45,
    'csnet_ns': 46, 'pop_2': 47, 'sunrpc': 48, 'uucp_path': 49, 'netbios_ns': 50,
    'netbios_ssn': 51, 'netbios_dgm': 52, 'sql_net': 53, 'vmnet': 54, 'bgp': 55,
    'Z39_50': 56, 'ldap': 57, 'netstat': 58, 'urh_i': 59, 'X11': 60, 'urp_i': 61,
    'pm_dump': 62, 'tftp_u': 63, 'tim_i': 64, 'red_i': 65, 'icmp': 66,
    'http_2784': 67, 'harvest': 68, 'aol': 69
}

FLAGS = {
    'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 'S1': 6,
    'S2': 7, 'RSTOS0': 8, 'S3': 9, 'OTH': 10
}


def load_nslkdd(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NSL-KDD dataset."""
    data_path = Path(data_path)
    
    train_file = data_path / 'KDDTrain+.txt'
    test_file = data_path / 'KDDTest+.txt'
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    print(f"Loading training data from {train_file}...")
    X_train, y_train = _load_file(train_file)
    
    print(f"Loading test data from {test_file}...")
    X_test, y_test = _load_file(test_file)
    
    return X_train, y_train, X_test, y_test


def _load_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single NSL-KDD file."""
    X_list = []
    y_list = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 42:
                continue
            
            features = []
            
            # Numeric features
            for i in [0] + list(range(4, 41)):
                try:
                    features.append(float(parts[i]))
                except (ValueError, IndexError):
                    features.append(0.0)
            
            # Categorical features
            features.append(PROTOCOLS.get(parts[1].lower(), len(PROTOCOLS)))
            features.append(SERVICES.get(parts[2].lower(), len(SERVICES)))
            features.append(FLAGS.get(parts[3].upper(), len(FLAGS)))
            
            X_list.append(features)
            
            attack_type = parts[41].lower().replace('.', '')
            y_list.append(ATTACK_CATEGORIES.get(attack_type, 0))
    
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def preprocess_data(X_train: np.ndarray, X_test: np.ndarray,
                    normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess features."""
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    if normalize:
        X_min = X_train.min(axis=0)
        X_max = X_train.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0
        
        X_train = (X_train - X_min) / X_range
        X_test = (X_test - X_min) / X_range
        
        X_train = np.clip(X_train, 0, 1)
        X_test = np.clip(X_test, 0, 1)
    
    return X_train, X_test


def get_dataset_statistics(y: np.ndarray) -> Dict:
    """Get dataset class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    names = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    stats = {'total': total, 'classes': {}}
    
    for c, count in zip(unique, counts):
        name = names[c] if c < len(names) else f'class_{c}'
        stats['classes'][name] = {'count': int(count), 'percentage': count / total * 100}
    
    return stats


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/NSL-KDD/'
    
    try:
        X_train, y_train, X_test, y_test = load_nslkdd(data_path)
        X_train, X_test = preprocess_data(X_train, X_test)
        
        print(f"\nDataset loaded!")
        print(f"Training: {len(X_train):,} samples")
        print(f"Test: {len(X_test):,} samples")
        
        print("\nTraining distribution:")
        for name, info in get_dataset_statistics(y_train)['classes'].items():
            print(f"  {name}: {info['count']:,} ({info['percentage']:.1f}%)")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nDownload from: https://www.unb.ca/cic/datasets/nsl.html")
