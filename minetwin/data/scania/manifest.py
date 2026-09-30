import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

REQUIRED_FILES = (
    "train_operational_readouts.csv",
    "train_specifications.csv",
    "train_tte.csv",
    "validation_operational_readouts.csv",
    "validation_specifications.csv",
    "validation_labels.csv",
    "test_operational_readouts.csv",
    "test_specifications.csv",
)
OPTIONAL_FILES = ("test_labels.csv",)


@dataclass(frozen=True)
class FileFingerprint:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ScaniaManifest:
    version: str
    doi: str
    source: str
    license: str
    created_at: str
    files: tuple[FileFingerprint, ...]

    @classmethod
    def build(cls, root: Path, version: str = "3") -> "ScaniaManifest":
        root = Path(root)
        missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Faltan archivos SCANIA requeridos: " + ", ".join(missing)
            )
        names = [*REQUIRED_FILES]
        names.extend(name for name in OPTIONAL_FILES if (root / name).is_file())
        return cls(
            version=version,
            doi="10.5878/bnh5-ka77",
            source="https://researchdata.se/en/catalogue/dataset/2024-34/1",
            license="CC BY 4.0",
            created_at=datetime.now(UTC).isoformat(),
            files=tuple(_fingerprint(root / name) for name in names),
        )

    @classmethod
    def read(cls, path: Path) -> "ScaniaManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            version=payload["version"],
            doi=payload["doi"],
            source=payload["source"],
            license=payload["license"],
            created_at=payload["created_at"],
            files=tuple(FileFingerprint(**item) for item in payload["files"]),
        )

    def verify(self, root: Path) -> None:
        root = Path(root)
        for expected in self.files:
            actual = _fingerprint(root / expected.name)
            if actual != expected:
                raise ValueError(f"El archivo {expected.name} no coincide con el manifiesto.")

    def write(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _fingerprint(path: Path) -> FileFingerprint:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return FileFingerprint(path.name, path.stat().st_size, digest.hexdigest())
