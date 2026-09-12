# Workflow

## Sequence Diagram — end-to-end run (video)

```mermaid
sequenceDiagram
    participant U as User (CLI)
    participant M as main.py
    participant P as Preprocessing
    participant PD as PersonDetector
    participant PP as PPEDetector
    participant T as IouTracker
    participant R as RuleEngine
    participant Z as ZoneCheck
    participant E as EvidenceExtractor
    participant Rp as Reporter

    U->>M: python -m src.main --input video.mp4 --mode video
    M->>M: build_config() validates args
    M->>PD: load person model
    M->>PP: load PPE model
    loop each video frame
        M->>P: read + resize frame
        P-->>M: Frame(image, index, timestamp)
        M->>PD: detect(frame)
        PD-->>M: [Detection(person)]
        M->>PP: detect(frame)
        PP-->>M: [Detection(helmet/head)]
        M->>T: update(person_boxes)
        T-->>M: [(track_id, bbox)]
        M->>R: evaluate_helmet_compliance(tracked, ppe)
        R-->>M: [ViolationEvent]
        M->>Z: check_restricted_zones(tracked, zones)
        Z-->>M: [ViolationEvent]
        alt violation found
            M->>E: capture_evidence(frame, event)
            E-->>M: EvidenceRecord
        end
        M->>M: draw annotated frame, append to output video
    end
    M->>Rp: write_json_report / write_csv_report / print_text_summary
    Rp-->>U: report.json, violations.csv, terminal summary
```

## Use Case Diagram

```mermaid
flowchart TB
    Officer((Safety Officer / Evaluator))
    Officer --> UC1[Run detection on image]
    Officer --> UC2[Run detection on video]
    Officer --> UC3[Configure restricted zones]
    Officer --> UC4[Review annotated output]
    Officer --> UC5[Review JSON/CSV report]
    Officer --> UC6[Review evidence images]
    Officer --> UC7[Retrain PPE model on new data]
```

## Class / Component Diagram

```mermaid
classDiagram
    class AppConfig {
        +Path input_path
        +Path output_dir
        +str mode
        +float confidence
        +list~ZoneConfig~ zones
    }
    class PersonDetector {
        +detect(image) list~Detection~
    }
    class PPEDetector {
        +detect(image) list~Detection~
    }
    class Detection {
        +bbox
        +confidence
        +class_name
    }
    class IouTracker {
        +update(boxes) list~tuple~
    }
    class ViolationEvent {
        +track_id
        +violation_type
        +confidence
    }
    class EvidenceRecord {
        +evidence_path
        +violation_type
    }
    class RunStats {
        +evidence: list~EvidenceRecord~
        +to_dict() dict
    }

    PersonDetector ..> Detection
    PPEDetector ..> Detection
    IouTracker ..> AppConfig
    ViolationEvent <.. PersonDetector
    ViolationEvent <.. PPEDetector
    EvidenceRecord ..> ViolationEvent
    RunStats o-- EvidenceRecord
```

No ER diagram is included — this project has no database or persistent
relational storage. Reports are stateless files written per run.
