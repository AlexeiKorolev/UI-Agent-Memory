"""Quick GPU check: start vLLM with the controller model and decode one real controller prompt (no cache writes)."""
import sys, time
from src.controller import EpisodeState, write_parts, extract_json, validate_write
from src.data import sample
from src.vlm import greedy, make_llm, to_messages
t0 = time.time()
mml = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
llm = make_llm("controller", max_model_len=mml, max_images=1, gpu_mem=0.95)
print("engine up in", round(time.time() - t0), "s", flush=True)
st = EpisodeState.__new__(EpisodeState)
from src.data import load_episode
e = sample(1)[0]; st.eid = e; st.ep = load_episode(e); st.entries = []
out = llm.chat([to_messages(write_parts(st, 2))], greedy(256), use_tqdm=False)[0].outputs[0].text
print("RAW:", out)
print("PARSED:", validate_write(extract_json(out)))
print("ENGINE_CHECK_OK")
