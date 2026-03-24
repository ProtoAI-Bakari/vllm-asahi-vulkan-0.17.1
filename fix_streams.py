#!/usr/bin/env python3
"""Fix torch.cuda.stream() calls to handle None streams for Vulkan."""
import re

filepath = "vllm/v1/worker/gpu_model_runner.py"
with open(filepath, "r") as f:
    content = f.read()

# Fix 1: Lines 233-237 (sample_batched_tokens)
# Change: with torch.cuda.stream(async_output_copy_stream):
# To: if async_output_copy_stream is not None:
#         with torch.cuda.stream(async_output_copy_stream):
old1 = """        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            if os.environ.get('VLLM_PLATFORM') != 'vulkan':
                async_output_copy_stream.wait_stream(default_stream)"""

new1 = """        default_stream = torch.cuda.current_stream()
        if async_output_copy_stream is not None:
            with torch.cuda.stream(async_output_copy_stream):
                if os.environ.get('VLLM_PLATFORM') != 'vulkan':
                    async_output_copy_stream.wait_stream(default_stream)"""

content = content.replace(old1, new1, 1)

# Fix 2: Lines 343-347 (process_pooler_output)
old2 = """        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            if os.environ.get('VLLM_PLATFORM') != 'vulkan':
                async_output_copy_stream.wait_stream(default_stream)
            self._model_runner_output.pooler_output = _copy_pooler_output_to_cpu("""

new2 = """        default_stream = torch.cuda.current_stream()
        if async_output_copy_stream is not None:
            with torch.cuda.stream(async_output_copy_stream):
                if os.environ.get('VLLM_PLATFORM') != 'vulkan':
                    async_output_copy_stream.wait_stream(default_stream)
            self._model_runner_output.pooler_output = _copy_pooler_output_to_cpu("""

content = content.replace(old2, new2, 1)

# Fix 3: Lines 4015-4021 (_copy_draft_token_ids_to_cpu)
old3 = """        default_stream = torch.cuda.current_stream()
        num_reqs = draft_token_ids.shape[0]
        with torch.cuda.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                # Trigger async copy of draft token ids to cpu.
                if os.environ.get('VLLM_PLATFORM') != 'vulkan':
                    self.draft_token_ids_copy_stream.wait_stream(default_stream)"""

new3 = """        default_stream = torch.cuda.current_stream()
        num_reqs = draft_token_ids.shape[0]
        if self.draft_token_ids_copy_stream is not None:
            with torch.cuda.stream(self.draft_token_ids_copy_stream):
                if not zeros_only:
                    # Trigger async copy of draft token ids to cpu.
                    if os.environ.get('VLLM_PLATFORM') != 'vulkan':
                        self.draft_token_ids_copy_stream.wait_stream(default_stream)"""

content = content.replace(old3, new3, 1)

# Fix 4: Lines 4048-4053 (_copy_valid_sampled_token_count_to_cpu)
old4 = """        default_stream = torch.cuda.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
            if os.environ.get('VLLM_PLATFORM') != 'vulkan':
                self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore"""

new4 = """        default_stream = torch.cuda.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        if self.valid_sampled_token_count_copy_stream is not None:
            with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
                if os.environ.get('VLLM_PLATFORM') != 'vulkan':
                    self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore"""

content = content.replace(old4, new4, 1)

with open(filepath, "w") as f:
    f.write(content)

print("Fixed all torch.cuda.stream() calls to handle None streams")