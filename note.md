# 1. รันการแปลงชุดข้อมูลเต็ม 700 ชั่วโมง
uv run python -m thai_audio_jepa_asr.cli.prepare_data --config configs/data/porjai.yaml --max-samples 0

# 2. ทำ Thai SSL Continued Pretraining
uv run python -m thai_audio_jepa_asr.cli.adapt_ssl --config configs/experiment/audio_jepa_thai_ssl.yaml

# 3. เทรน Supervised ASR บนชุดข้อมูลฉบับเต็ม
uv run python -m thai_audio_jepa_asr.cli.train_asr --config configs/experiment/audio_jepa_ctc.yaml

# 4. เทรนการศึกษา Low-Resource (เช่น 10%, 25%, 50%)
uv run python -m thai_audio_jepa_asr.cli.train_asr --config configs/experiment/low_resource_10.yaml

# 5. สรุปตารางเปรียบเทียบผลการทดลองทั้งหมด
uv run python -m thai_audio_jepa_asr.cli.compare --runs-dir outputs/runs
