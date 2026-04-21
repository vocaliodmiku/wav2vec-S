import glob
import os

import soundfile as sf
import torch
import torch.nn as nn
from jiwer import wer
from tqdm import tqdm
from transformers import Wav2Vec2Processor

from wav2vec_s.modeling_wav2vec_s import Wav2VecSModel, Wav2VecSPreTrainedModel


class Wav2VecSForCTC(Wav2VecSPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2VecSModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs[0])
        logits = self.lm_head(hidden_states)
        return logits


model_path = "biaofu-xmu/wav2vec-S-Large-ft-960h"

processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2VecSForCTC.from_pretrained(model_path)
model.to(torch.bfloat16).cuda()
model.eval()
model.wav2vec2.encoder.main_context = 8 # [160ms,640ms] m
model.wav2vec2.encoder.right_context = 2 # [80ms,320ms] r; r <= m/2
print("model.wav2vec2.encoder.context_type:", model.wav2vec2.encoder.context_type)
print("model.wav2vec2.encoder.main_context:", model.wav2vec2.encoder.main_context)
print("model.wav2vec2.encoder.right_context:", model.wav2vec2.encoder.right_context)

# load LibriSpeech test-clean from a local directory
librispeech_root = "/scratch/jsm04005/fie24002/LibriSpeech/LibriSpeech/test-clean"

samples = []
for trans_file in glob.glob(os.path.join(librispeech_root, "*", "*", "*.trans.txt")):
    chapter_dir = os.path.dirname(trans_file)
    with open(trans_file) as f:
        for line in f:
            utt_id, text = line.strip().split(" ", 1)
            samples.append((os.path.join(chapter_dir, utt_id + ".flac"), text))

predictions = []
references = []

with torch.no_grad():
    for flac_path, reference in tqdm(samples):
        waveform, sampling_rate = sf.read(flac_path)

        inputs = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
        input_values = inputs.input_values.to("cuda").to(torch.bfloat16)

        logits = model(input_values)
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)[0]

        predictions.append(transcription)
        references.append(reference)

score = wer(references, predictions)
print(f"WER on LibriSpeech test-clean: {score * 100:.2f}%")
