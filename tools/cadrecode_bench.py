import json
import os
import sys
import tempfile
import traceback

import numpy as np

os.environ.setdefault("P2F_SQUARE_MM", "30")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
N_POINTS = 256


def farthest_points(pts, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d)); idx.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return pts[idx]


def normalised(mesh):
    m = mesh.copy()
    m.apply_translation(-(m.bounds[0] + m.bounds[1]) / 2)
    m.apply_scale(2.0 / max(float(np.max(m.extents)), 1e-9))
    return m


def cloud_from_mesh(mesh, n=N_POINTS):
    import trimesh
    v, _ = trimesh.sample.sample_surface(normalised(mesh), 8192)
    return farthest_points(np.asarray(v, float), n)


def load_model():
    import torch
    from torch import nn
    from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Model, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast

    class FourierPointEncoder(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
            self.register_buffer('frequencies', frequencies, persistent=False)
            self.projection = nn.Linear(51, hidden_size)

        def forward(self, points):
            x = points
            x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
            x = torch.cat((points, x.sin(), x.cos()), dim=-1)
            return self.projection(x)

    class CADRecode(Qwen2ForCausalLM):
        def __init__(self, config):
            PreTrainedModel.__init__(self, config)
            self.model = Qwen2Model(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            torch.set_default_dtype(torch.float32)
            self.point_encoder = FourierPointEncoder(config.hidden_size)
            torch.set_default_dtype(torch.bfloat16)

        def forward(self, input_ids=None, attention_mask=None, point_cloud=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                    labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, cache_position=None):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            if past_key_values is None or past_key_values.get_seq_length() == 0:
                inputs_embeds = self.model.embed_tokens(input_ids)
                point_embeds = self.point_encoder(point_cloud).bfloat16()
                inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
                attention_mask[attention_mask == -1] = 1
                input_ids = None
                position_ids = None
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states, return_dict=return_dict, cache_position=cache_position)
            logits = self.lm_head(outputs[0]).float()
            if not return_dict:
                return (logits,) + outputs[1:]
            return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=outputs.past_key_values,
                                          hidden_states=outputs.hidden_states, attentions=outputs.attentions)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
            model_inputs['point_cloud'] = kwargs['point_cloud']
            return model_inputs

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B', pad_token='<|im_end|>', padding_side='left')
    attn = os.environ.get("CADRECODE_ATTN", "sdpa")
    model = CADRecode.from_pretrained('filapro/cad-recode-v1.5', torch_dtype='auto', attn_implementation=attn).eval()
    return tokenizer, model.to("cuda" if torch.cuda.is_available() else "cpu")


def generate(tokenizer, model, cloud):
    import torch
    input_ids = [tokenizer.pad_token_id] * len(cloud) + [tokenizer('<|im_start|>')['input_ids'][0]]
    attention_mask = [-1] * len(cloud) + [1]
    with torch.no_grad():
        ids = model.generate(input_ids=torch.tensor(input_ids).unsqueeze(0).to(model.device),
                             attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(model.device),
                             point_cloud=torch.tensor(cloud.astype(np.float32)).unsqueeze(0).to(model.device),
                             max_new_tokens=768, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(ids)[0].split('<|im_start|>')[-1].replace('<|im_end|>', '').replace('<|endoftext|>', '')


def execute(code):
    import cadquery as cq
    import trimesh
    g = {"cq": cq}
    exec(code, g)
    v, f = g["r"].val().tessellate(0.001, 0.1)
    return trimesh.Trimesh([(p.x, p.y, p.z) for p in v], f)


def hull_for(part):
    import cv2, trimesh
    import board_bench as B
    import test_board_path as T
    from photo2fcstd import bench, carve, make_target
    m = B.rest_on_largest_face(trimesh.load(bench.truth_of(part)))
    m.apply_scale(B.PART_MM / float(np.max(m.extents[:2])))
    W, Hh = make_target.COLS * make_target.SQUARE_MM, make_target.ROWS * make_target.SQUARE_MM
    centre = np.array([W / 2, Hh / 2, 0.0])
    m.apply_translation(centre - np.array([m.bounds[:, 0].mean(), m.bounds[:, 1].mean(), m.bounds[0][2]]))
    K = T._camera(B.SIZE[0], B.SIZE[1], B.FOCAL); d = tempfile.mkdtemp(prefix="cr_"); photos = []
    for k, (az, el) in enumerate(B.VIEWS):
        eye = centre + 520.0 * np.array([np.cos(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(el))])
        R, t = T._look_at(eye, target=centre); p = os.path.join(d, "v%d.jpg" % k)
        cv2.imwrite(p, cv2.cvtColor(B.render_mesh(m, R, t, K), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); photos.append(p)
    carved = carve.from_photos(photos, voxel_mm=B.VOXEL)
    import shutil; shutil.rmtree(d, ignore_errors=True)
    return carve.mesh_of(carved), m


def main(parts):
    import trimesh
    from photo2fcstd import bench, score
    tokenizer, model = load_model()
    rows = []
    for k, part in enumerate(parts, 1):
        row = {"part": part}
        try:
            truth = trimesh.load(bench.truth_of(part))
            for name, src in (("truth", truth), ("hull", None)):
                if src is None:
                    hull_mesh, oriented = hull_for(part)
                    row["hull_mesh_iou"] = round(float(score.best_iou(oriented, hull_mesh)[0]), 4)
                    src = hull_mesh
                code = generate(tokenizer, model, cloud_from_mesh(src))
                try:
                    mesh = execute(code)
                    row[name + "_iou"] = round(float(score.best_iou(truth, mesh)[0]), 4)
                    row[name + "_code"] = code[:600]
                except Exception as e:
                    row[name + "_iou"] = None; row[name + "_error"] = str(e)[:100]
        except Exception as e:
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:120]); row["trace"] = traceback.format_exc()[-400:]
        rows.append(row)
        if k % 5 == 0 or k == len(parts):
            print("progress %d/%d" % (k, len(parts)), flush=True)
            json.dump(rows, open(os.path.join(ROOT, "runs", "cadrecode_bench.json"), "w"), indent=1)
    ok = [r for r in rows if r.get("truth_iou") is not None]; hk = [r for r in rows if r.get("hull_iou") is not None]
    print("CADRECODE n=%d | from truth points: valid %d, 3D IoU mean %.3f, >=0.8 %d | from board hull points: valid %d, 3D IoU mean %.3f, >=0.8 %d | hull mesh itself: %.3f"
          % (len(rows), len(ok), np.mean([r["truth_iou"] for r in ok]) if ok else 0, sum(r["truth_iou"] >= 0.8 for r in ok),
             len(hk), np.mean([r["hull_iou"] for r in hk]) if hk else 0, sum(r["hull_iou"] >= 0.8 for r in hk),
             np.mean([r["hull_mesh_iou"] for r in rows if "hull_mesh_iou" in r]) if any("hull_mesh_iou" in r for r in rows) else 0))


if __name__ == "__main__":
    main(sys.argv[1:] or open(os.path.join(ROOT, "runs", "bench_parts.txt")).read().split())
