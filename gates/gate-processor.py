"""CPU-only, offline GLM processor acceptance; no weights or remote code loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import os


def check_output(processor, rendered, images):
    import torch

    raw = processor.tokenizer(rendered, return_tensors='pt')['input_ids']
    token = processor.tokenizer.convert_tokens_to_ids('<|image|>')
    assert token == processor.image_token_id == 154854, 'checkpoint image token changed'
    assert int((raw == token).sum()) == len(images), 'placeholder count'
    output = processor(text=[rendered], images=images, padding=True,
                       return_tensors='pt', device='cpu')
    ids, pixels, grids = (output[k] for k in ('input_ids', 'pixel_values', 'image_grid_thw'))
    assert grids.shape == (len(images), 3) and bool((grids > 0).all()), 'image grids'
    patches = grids.prod(dim=1)
    merge = processor.image_processor.merge_size ** 2
    assert bool((patches % merge == 0).all()), 'nonintegral merge'
    image_tokens = int((patches // merge).sum())
    assert int((ids == token).sum()) == image_tokens > len(images), 'image token accounting'
    assert ids.numel() == raw.numel() - len(images) + image_tokens, 'unexpanded image IDs'
    assert pixels.ndim == 2 and pixels.shape[0] == int(patches.sum()), 'pixel/grid accounting'
    assert pixels.shape[1] == (3 * processor.image_processor.temporal_patch_size
                               * processor.image_processor.patch_size ** 2), 'pixel patch width'
    assert bool(torch.isfinite(pixels).all()), 'nonfinite pixels'
    return output


def probe(processor):
    import torch
    from PIL import Image
    from transformers import Glm5NextProcessor, Glm5NextImageProcessor, Glm5NextVideoProcessor

    assert isinstance(processor, Glm5NextProcessor), type(processor).__name__
    assert isinstance(processor.image_processor, Glm5NextImageProcessor)
    assert isinstance(processor.video_processor, Glm5NextVideoProcessor)
    template = processor.chat_template or processor.tokenizer.chat_template
    if isinstance(template, dict):
        template = template['default']
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Describe both images.'},
                                             {'type': 'image'}, {'type': 'image'}]}]
    rendered = processor.tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False, add_generation_prompt=True)
    red, blue = (Image.new('RGB', (512, 512), color) for color in ('red', 'blue'))
    rb = check_output(processor, rendered, [red, blue])
    br = check_output(processor, rendered, [blue, red])
    rr = check_output(processor, rendered, [red, red])
    assert torch.equal(rb['input_ids'], br['input_ids'])
    assert torch.equal(rb['image_grid_thw'], br['image_grid_thw'])
    size = int(rb['image_grid_thw'][0].prod())
    assert not torch.equal(rb['pixel_values'][:size], rb['pixel_values'][size:]), 'images silently collapsed'
    assert torch.equal(rb['pixel_values'][:size], br['pixel_values'][size:]), 'red order lost'
    assert torch.equal(rb['pixel_values'][size:], br['pixel_values'][:size]), 'blue order lost'
    assert torch.equal(rb['pixel_values'][:size], rr['pixel_values'][:size])
    assert torch.equal(rr['pixel_values'][:size], rr['pixel_values'][size:])
    return rb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--metadata', type=Path, default=Path(__file__).with_name('processor-metadata.json'))
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    # Force native TorchCodec/FFmpeg loading on CPU, not just lazy class presence.
    from torchcodec.decoders import VideoDecoder
    assert callable(VideoDecoder), 'video decoder prerequisite missing'
    from transformers import AutoProcessor
    from sglang.srt.utils.hf_transformers_utils import get_processor
    import torch

    hashes = json.loads(args.metadata.read_text())
    with tempfile.TemporaryDirectory(prefix='glm-metadata-') as directory:
        for name, digest in hashes.items():
            assert Path(name).name == name
            data = (args.checkpoint / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest, ('checkpoint metadata drift', name)
            (Path(directory) / name).write_bytes(data)
        run(Path(directory), AutoProcessor, get_processor, torch, hashes)


def run(checkpoint, AutoProcessor, get_processor, torch, hashes):
    # Both real selection paths must work: imports alone previously passed while
    # get_processor returned a tokenizer and silently discarded the images.
    direct = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=False, local_files_only=True)
    selected = get_processor(str(checkpoint), trust_remote_code=False, local_files_only=True)
    expected = probe(direct)
    actual = probe(selected)
    for key in ('input_ids', 'pixel_values', 'image_grid_thw'):
        assert torch.equal(expected[key], actual[key]), ('selection mismatch', key)
    with tempfile.TemporaryDirectory(prefix='glm-processor-') as directory:
        selected.save_pretrained(directory)
        restored = AutoProcessor.from_pretrained(directory, trust_remote_code=False, local_files_only=True)
        reloaded = probe(restored)
        for key in ('input_ids', 'pixel_values', 'image_grid_thw'):
            assert torch.equal(actual[key], reloaded[key]), ('serialization mismatch', key)
    # Negative control: the real gate must reject tokenizer-only fallback.
    try:
        probe(selected.tokenizer)
    except (AssertionError, AttributeError):
        pass
    else:
        raise AssertionError('tokenizer-only negative control passed')
    print('GLM_PROCESSOR_PASS', json.dumps({'metadata_sha256': hashes,
          'input_ids': list(actual['input_ids'].shape),
          'pixel_values': list(actual['pixel_values'].shape),
          'image_grid_thw': actual['image_grid_thw'].tolist()}), flush=True)


if __name__ == '__main__':
    main()
