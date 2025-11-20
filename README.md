# Apex_AI

## Setup

### 1. Create and activate a virtual environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# OR
.venv\Scripts\activate     # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure OpenAI API key

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Analyze pre-captured screenshots + logs
```bash
python sample_sum.py \
  --log_file sample_data/collection_20251119_164558/log_test.txt \
  --frames_folder sample_data/collection_20251119_164558/frames \
  --model gpt-5 \
  --output my_gameplay_summary.txt
```

### Analyze recorded video (basic)
```bash
python sum_test.py
```

### Analyze recorded video + JSON logs
```bash
python sum_test_toon.py \
  --video sample_data/collection_20251119_165050/gameplay_recording.mp4 \
  --log sample_data/collection_20251119_165050/log_test.txt \
  --output summary.txt
```

