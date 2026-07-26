"""
generate_data.py

Generates a realistic, synthetic enterprise authentication log dataset for
AI-powered behavioural anomaly detection (Honeywell hackathon).

The dataset simulates login events for three kinds of entities (human users,
service accounts, and edge devices), each with its own stable behavioural
profile (preferred hours, country, IP range, browser, OS, auth method,
device fingerprint, and resources). On top of ~97% normal traffic, seven
distinct attack patterns are injected to create labelled anomalies.

Run with:
    python generate_data.py

Only standard-library modules plus pandas, numpy, and faker are used.
"""

from __future__ import annotations

import random
import uuid
import ipaddress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from faker import Faker

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #

RANDOM_SEED = 42
TOTAL_EVENTS = 10_000
ATTACK_FRACTION = 0.03  # ~3% of the dataset is anomalous
DAYS_OF_HISTORY = 90

COUNTRIES = [
    "India", "USA", "Germany", "Japan", "Singapore",
    "Canada", "Australia", "Brazil", "France", "UK",
]

AUTH_METHODS = ["password", "MFA", "certificate", "token", "biometric"]

OPERATING_SYSTEMS = ["Windows", "Linux", "macOS", "Ubuntu", "Android", "Firmware"]

BROWSERS = ["Chrome", "Edge", "Firefox", "Safari", "API"]

RESOURCES = [
    "Payroll", "HR", "Finance", "Email", "VPN", "Database",
    "Production Server", "SCADA", "PLC", "Cloud Dashboard", "IoT Gateway",
]

# Resources typically reachable by each entity type (used to build profiles).
ENTITY_RESOURCE_POOL: Dict[str, List[str]] = {
    "user": ["Email", "VPN", "Payroll", "HR", "Finance", "Database", "Cloud Dashboard"],
    "service_account": ["Database", "Production Server", "Cloud Dashboard", "VPN"],
    "edge_device": ["SCADA", "PLC", "IoT Gateway", "Production Server"],
}

# Mapping from resource to a realistic command sequence (multiple options).
COMMAND_SEQUENCES: Dict[str, List[str]] = {
    "Email": ["login>email"],
    "VPN": ["login>vpn"],
    "Database": ["login>vpn>database", "login>server>database"],
    "Payroll": ["login>payroll"],
    "HR": ["login>hr"],
    "Finance": ["login>finance"],
    "Production Server": ["login>server"],
    "SCADA": ["login>scada"],
    "PLC": ["login>plc"],
    "Cloud Dashboard": ["login>cloud"],
    "IoT Gateway": ["login>cloud>iot", "login>iot"],
}

ATTACK_TYPES = [
    "Brute Force",
    "Impossible Travel",
    "Credential Stuffing",
    "Device Spoofing",
    "Lateral Movement",
    "Low and Slow Exfiltration",
    "Insider Drift",
]

fake = Faker()


# --------------------------------------------------------------------------- #
# UserProfile
# --------------------------------------------------------------------------- #

@dataclass
class UserProfile:
    """Behavioural profile for a single enterprise entity.

    Captures the stable, "normal" behaviour of a user, service account, or
    edge device so that normal events can be sampled consistently and
    attacks can be injected as deviations from this baseline.
    """

    entity_id: str
    entity_type: str
    normal_hours: List[int]
    country: str
    ip_network: ipaddress.IPv4Network
    browser: str
    operating_system: str
    auth_method: str
    device_fingerprint: str
    resources: List[str]

    def sample_hour(self) -> int:
        """Return a login hour, mostly within normal working hours."""
        if random.random() < 0.9:
            return random.choice(self.normal_hours)
        return random.randint(0, 23)

    def sample_ip(self) -> str:
        """Return an IP address drawn from this entity's preferred subnet."""
        hosts = list(self.ip_network.hosts())
        return str(random.choice(hosts))

    def sample_resource(self) -> str:
        """Return a resource from this entity's normal resource pool."""
        return random.choice(self.resources)

    @staticmethod
    def command_sequence_for(resource: str) -> str:
        """Return a plausible command sequence string for a given resource."""
        options = COMMAND_SEQUENCES.get(resource, ["login"])
        return random.choice(options)


# --------------------------------------------------------------------------- #
# Profile factory
# --------------------------------------------------------------------------- #

class ProfileFactory:
    """Builds a population of UserProfile objects for all entity types."""

    ENTITY_COUNTS = {
        "user": 40,
        "service_account": 10,
        "edge_device": 10,
    }

    @classmethod
    def build_profiles(cls) -> List[UserProfile]:
        """Create and return the full list of entity behavioural profiles."""
        profiles: List[UserProfile] = []
        for entity_type, count in cls.ENTITY_COUNTS.items():
            for i in range(count):
                profiles.append(cls._build_single_profile(entity_type, i))
        return profiles

    @staticmethod
    def _build_single_profile(entity_type: str, index: int) -> UserProfile:
        """Construct one profile for the given entity type."""
        prefix = {"user": "user", "service_account": "svc", "edge_device": "dev"}[entity_type]
        entity_id = f"{prefix}_{index + 1:04d}"

        # Normal working hours differ slightly by entity type.
        if entity_type == "user":
            normal_hours = list(range(8, 19))  # daytime office hours
        elif entity_type == "service_account":
            normal_hours = list(range(0, 24))  # can run any time, batch jobs
        else:  # edge_device
            normal_hours = list(range(0, 24))  # devices operate continuously

        # Assign a private-range /24 network as the "preferred" IP range.
        octet_a = random.choice([10, 172, 192])
        if octet_a == 10:
            base = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.0/24"
        elif octet_a == 172:
            base = f"172.{random.randint(16, 31)}.{random.randint(0, 255)}.0/24"
        else:
            base = f"192.168.{random.randint(0, 255)}.0/24"
        ip_network = ipaddress.ip_network(base)

        resources = random.sample(
            ENTITY_RESOURCE_POOL[entity_type],
            k=min(3, len(ENTITY_RESOURCE_POOL[entity_type])),
        )

        return UserProfile(
            entity_id=entity_id,
            entity_type=entity_type,
            normal_hours=normal_hours,
            country=random.choice(COUNTRIES),
            ip_network=ip_network,
            browser=random.choice(BROWSERS),
            operating_system=random.choice(OPERATING_SYSTEMS),
            auth_method=random.choice(AUTH_METHODS),
            device_fingerprint=str(uuid.uuid4()),
            resources=resources,
        )


# --------------------------------------------------------------------------- #
# EnterpriseDataGenerator
# --------------------------------------------------------------------------- #

class EnterpriseDataGenerator:
    """Generates the baseline (normal) portion of the login event dataset."""

    def __init__(self, profiles: List[UserProfile], start_date: datetime) -> None:
        """Store profiles and the earliest timestamp events can take."""
        self.profiles = profiles
        self.start_date = start_date

    def random_timestamp(self, profile: UserProfile) -> datetime:
        """Return a random timestamp within the history window that respects
        the entity's normal login hours most of the time."""
        day_offset = random.randint(0, DAYS_OF_HISTORY - 1)
        day = self.start_date + timedelta(days=day_offset)
        hour = profile.sample_hour()
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        return day.replace(hour=hour, minute=minute, second=second, microsecond=0)

    def generate_normal_event(self, profile: UserProfile) -> Dict[str, Any]:
        """Create a single normal (non-anomalous) login event for a profile."""
        resource = profile.sample_resource()
        session_duration = int(np.random.normal(loc=600, scale=200))
        session_duration = max(30, session_duration)

        return {
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": self.random_timestamp(profile),
            "source_ip": profile.sample_ip(),
            "geo_location": profile.country,
            "resource_accessed": resource,
            "auth_method": profile.auth_method,
            "session_duration": session_duration,
            "command_sequence": UserProfile.command_sequence_for(resource),
            "device_fingerprint": profile.device_fingerprint,
            "browser": profile.browser,
            "operating_system": profile.operating_system,
            "login_result": "success" if random.random() < 0.97 else "fail",
            "risk_score": int(np.clip(np.random.normal(loc=15, scale=8), 0, 39)),
            "attack_type": "Normal",
            "label": 0,
        }

    def generate_normal_events(self, count: int) -> List[Dict[str, Any]]:
        """Create `count` normal login events spread across all profiles."""
        events: List[Dict[str, Any]] = []
        for _ in range(count):
            profile = random.choice(self.profiles)
            events.append(self.generate_normal_event(profile))
        return events


# --------------------------------------------------------------------------- #
# AttackInjector
# --------------------------------------------------------------------------- #

class AttackInjector:
    """Generates anomalous login events for each supported attack pattern."""

    def __init__(self, profiles: List[UserProfile], start_date: datetime) -> None:
        """Store profiles and the earliest timestamp events can take."""
        self.profiles = profiles
        self.start_date = start_date

    # --- shared helpers ---------------------------------------------------- #

    def _random_base_time(self) -> datetime:
        """Return a random base timestamp within the history window."""
        day_offset = random.randint(0, DAYS_OF_HISTORY - 1)
        day = self.start_date + timedelta(days=day_offset)
        return day.replace(
            hour=random.randint(0, 23),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
            microsecond=0,
        )

    def _risk_score(self) -> int:
        """Return a high risk score typical of attack events."""
        return random.randint(70, 100)

    def _base_event(self, profile: UserProfile, attack_type: str) -> Dict[str, Any]:
        """Return a skeleton event pre-filled with profile defaults."""
        resource = profile.sample_resource()
        return {
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": self._random_base_time(),
            "source_ip": profile.sample_ip(),
            "geo_location": profile.country,
            "resource_accessed": resource,
            "auth_method": profile.auth_method,
            "session_duration": random.randint(30, 300),
            "command_sequence": UserProfile.command_sequence_for(resource),
            "device_fingerprint": profile.device_fingerprint,
            "browser": profile.browser,
            "operating_system": profile.operating_system,
            "login_result": "fail",
            "risk_score": self._risk_score(),
            "attack_type": attack_type,
            "label": 1,
        }

    # --- individual attack generators --------------------------------------- #

    def brute_force(self, profile: UserProfile, n_attempts: int = 8) -> List[Dict[str, Any]]:
        """Many failed logins, same IP, same account, very short interval."""
        events = []
        base_time = self._random_base_time()
        fixed_ip = profile.sample_ip()
        for i in range(n_attempts):
            event = self._base_event(profile, "Brute Force")
            event["timestamp"] = base_time + timedelta(seconds=5 * i)
            event["source_ip"] = fixed_ip
            event["login_result"] = "fail" if i < n_attempts - 1 else "success"
            events.append(event)
        return events

    def impossible_travel(self, profile: UserProfile) -> List[Dict[str, Any]]:
        """Same user logging in from two different countries within minutes."""
        base_time = self._random_base_time()
        other_country = random.choice([c for c in COUNTRIES if c != profile.country])

        event_1 = self._base_event(profile, "Impossible Travel")
        event_1["timestamp"] = base_time
        event_1["geo_location"] = profile.country
        event_1["login_result"] = "success"

        event_2 = self._base_event(profile, "Impossible Travel")
        event_2["timestamp"] = base_time + timedelta(minutes=random.randint(2, 10))
        event_2["geo_location"] = other_country
        event_2["source_ip"] = str(fake.ipv4_public())
        event_2["login_result"] = "success"

        return [event_1, event_2]

    def credential_stuffing(self, n_usernames: int = 10) -> List[Dict[str, Any]]:
        """One IP hammering many different usernames, mostly failed."""
        events = []
        attacker_ip = str(fake.ipv4_public())
        base_time = self._random_base_time()
        targets = random.sample(self.profiles, k=min(n_usernames, len(self.profiles)))
        for i, profile in enumerate(targets):
            event = self._base_event(profile, "Credential Stuffing")
            event["timestamp"] = base_time + timedelta(seconds=3 * i)
            event["source_ip"] = attacker_ip
            event["login_result"] = "success" if random.random() < 0.1 else "fail"
            events.append(event)
        return events

    def device_spoofing(self, profile: UserProfile) -> List[Dict[str, Any]]:
        """Known user logging in with an unfamiliar device and OS."""
        event = self._base_event(profile, "Device Spoofing")
        event["device_fingerprint"] = str(uuid.uuid4())
        event["operating_system"] = random.choice(
            [os for os in OPERATING_SYSTEMS if os != profile.operating_system]
        )
        event["login_result"] = random.choice(["success", "fail"])
        return [event]

    def lateral_movement(self, profile: UserProfile) -> List[Dict[str, Any]]:
        """User accesses resources it has never touched before."""
        unseen_resources = [r for r in RESOURCES if r not in profile.resources]
        chosen = random.sample(unseen_resources, k=min(3, len(unseen_resources)))
        events = []
        base_time = self._random_base_time()
        for i, resource in enumerate(chosen):
            event = self._base_event(profile, "Lateral Movement")
            event["timestamp"] = base_time + timedelta(minutes=10 * i)
            event["resource_accessed"] = resource
            event["command_sequence"] = UserProfile.command_sequence_for(resource)
            event["login_result"] = "success"
            events.append(event)
        return events

    def low_and_slow_exfiltration(self, profile: UserProfile, n_days: int = 5) -> List[Dict[str, Any]]:
        """Small off-hour accesses to sensitive resources spread over days."""
        sensitive_resources = ["Database", "Finance", "Payroll", "Production Server", "SCADA"]
        events = []
        base_day = self.start_date + timedelta(days=random.randint(0, DAYS_OF_HISTORY - n_days - 1))
        for d in range(n_days):
            event = self._base_event(profile, "Low and Slow Exfiltration")
            off_hour = random.choice([0, 1, 2, 3, 4, 23])
            event["timestamp"] = (base_day + timedelta(days=d)).replace(hour=off_hour)
            event["resource_accessed"] = random.choice(sensitive_resources)
            event["command_sequence"] = UserProfile.command_sequence_for(event["resource_accessed"])
            event["session_duration"] = random.randint(1800, 5400)  # long sessions
            event["login_result"] = "success"
            events.append(event)
        return events

    def insider_drift(self, profile: UserProfile, n_events: int = 6) -> List[Dict[str, Any]]:
        """Legitimate user slowly expanding into new resource territory."""
        events = []
        base_time = self._random_base_time()
        expanding_pool = [r for r in RESOURCES if r not in profile.resources]
        for i in range(n_events):
            event = self._base_event(profile, "Insider Drift")
            event["timestamp"] = base_time + timedelta(days=i)
            resource = random.choice(expanding_pool) if expanding_pool else profile.sample_resource()
            event["resource_accessed"] = resource
            event["command_sequence"] = UserProfile.command_sequence_for(resource)
            event["login_result"] = "success"
            # Risk score gradually creeps up as drift continues.
            event["risk_score"] = int(np.clip(70 + i * 3, 70, 100))
            events.append(event)
        return events

    # --- orchestration -------------------------------------------------------#

    def generate_attacks(self, total_attack_events: int) -> List[Dict[str, Any]]:
        """Generate a mix of all attack types until reaching the target count."""
        events: List[Dict[str, Any]] = []
        human_profiles = [p for p in self.profiles if p.entity_type == "user"]

        while len(events) < total_attack_events:
            attack_choice = random.choice(ATTACK_TYPES)
            profile = random.choice(human_profiles) if human_profiles else random.choice(self.profiles)

            if attack_choice == "Brute Force":
                events.extend(self.brute_force(profile, n_attempts=random.randint(6, 10)))
            elif attack_choice == "Impossible Travel":
                events.extend(self.impossible_travel(profile))
            elif attack_choice == "Credential Stuffing":
                events.extend(self.credential_stuffing(n_usernames=random.randint(8, 12)))
            elif attack_choice == "Device Spoofing":
                events.extend(self.device_spoofing(profile))
            elif attack_choice == "Lateral Movement":
                events.extend(self.lateral_movement(profile))
            elif attack_choice == "Low and Slow Exfiltration":
                events.extend(self.low_and_slow_exfiltration(profile, n_days=random.randint(3, 6)))
            elif attack_choice == "Insider Drift":
                events.extend(self.insider_drift(profile, n_events=random.randint(4, 7)))

        # Trim any overshoot so the final count matches the target exactly.
        return events[:total_attack_events]


# --------------------------------------------------------------------------- #
# Pipeline / orchestration
# --------------------------------------------------------------------------- #

def build_dataset() -> pd.DataFrame:
    """Build the complete, shuffled login-event dataset as a DataFrame."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    start_date = datetime.now() - timedelta(days=DAYS_OF_HISTORY)

    profiles = ProfileFactory.build_profiles()

    n_attack_events = int(TOTAL_EVENTS * ATTACK_FRACTION)
    n_normal_events = TOTAL_EVENTS - n_attack_events

    normal_generator = EnterpriseDataGenerator(profiles, start_date)
    normal_events = normal_generator.generate_normal_events(n_normal_events)

    injector = AttackInjector(profiles, start_date)
    attack_events = injector.generate_attacks(n_attack_events)

    all_events = normal_events + attack_events
    random.shuffle(all_events)

    df = pd.DataFrame(all_events)
    df.sort_values("timestamp", inplace=True)  # optional chronological aid
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # final shuffle

    # Reorder columns to match the required schema exactly.
    column_order = [
        "entity_id", "entity_type", "timestamp", "source_ip", "geo_location",
        "resource_accessed", "auth_method", "session_duration", "command_sequence",
        "device_fingerprint", "browser", "operating_system", "login_result",
        "risk_score", "attack_type", "label",
    ]
    return df[column_order]


def save_dataset(df: pd.DataFrame) -> Path:
    """Save the dataset to data/raw/login_logs.csv, creating folders as needed."""
    output_path = Path("data/raw/login_logs.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of the generated dataset to the console."""
    total = len(df)
    normal = int((df["label"] == 0).sum())
    attacks = int((df["label"] == 1).sum())

    print("Dataset generated successfully")
    print(f"Total records: {total}")
    print(f"Normal records: {normal}")
    print(f"Attack records: {attacks}")
    print("Attack distribution:")
    print(df.loc[df["label"] == 1, "attack_type"].value_counts().to_string())


def main() -> None:
    """Entry point: build, save, and summarize the synthetic dataset."""
    df = build_dataset()
    save_dataset(df)
    print_summary(df)


if __name__ == "__main__":
    main()