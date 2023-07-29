# %%
from neel.imports import *
from neel_plotly import *
model = HookedTransformer.from_pretrained("gelu-1l")
utils.test_prompt(" Nobel Peace", "Prize", model)
torch.set_grad_enabled(False)
# %%
NOBEL = model.to_single_token(" Nobel")
PEACE = model.to_single_token(" Peace")
PRIZE = model.to_single_token(" Prize")

logits, cache = model.run_with_cache(" Nobel Peace")
unembed_dir = model.tokens_to_residual_directions(" Peace")
resid_stack, labels = cache.get_full_resid_decomposition(apply_ln=True, pos_slice=-1, return_labels=True)

line(resid_stack @ unembed_dir, x=labels, title="DLA of components on clean input")

# %%
def show_neuron_dla(ni, sample_tokens=[], show_tail=True):
    neuron_vocab_df = nutils.create_vocab_df(model.W_out[0, ni] @ model.W_U, model=model)
    neuron_vocab_df["rank"] = np.arange(len(neuron_vocab_df))
    nutils.show_df(neuron_vocab_df.head(50))
    if show_tail:
        nutils.show_df(neuron_vocab_df.tail(20))
    for tok in sample_tokens:
        print(neuron_vocab_df.loc[model.to_single_token(tok)])
    return neuron_vocab_df
_ = show_neuron_dla(1899, ["ric"])

# %%
# %%
no_prefix_logits, no_prefix_cache = model.run_with_cache(" Peace")
no_prefix_neuron_acts = no_prefix_cache["post", 0][0, -1]

neuron_df = nutils.make_neuron_df(model.cfg.n_layers, model.cfg.d_mlp)
clean_neuron_acts = cache["post", 0][0, -1]
neuron_df["clean"] = clean_neuron_acts.tolist()
neuron_df["no_prefix"] = no_prefix_neuron_acts.tolist()
neuron_df["wdla"] = (model.W_out[0] @ model.W_U[:, PRIZE]).tolist()
neuron_df["clean_dla"] = (neuron_df["clean"] * neuron_df["wdla"]).tolist()
neuron_df["no_prefix_dla"] = (neuron_df["no_prefix"] * neuron_df["wdla"]).tolist()

# %%
d_vocab = model.cfg.d_vocab
full_vocab = model.to_str_tokens(np.arange(d_vocab))
vocab_df = pd.DataFrame({
    "token": full_vocab,
    "unique_token": nutils.process_tokens_index(np.arange(d_vocab), model),
    "index": torch.arange(d_vocab).tolist(),
})
vocab_df["has_space"] = vocab_df.token.str.contains(" ")
vocab_df["start_cap"] = [len(s.strip())>0 and s.strip()[0].isupper() for s in vocab_df.token]
vocab_df["is_capital"] = vocab_df["has_space"] & vocab_df["start_cap"]
vocab_df["is_nobel"] = vocab_df.token == " Nobel"
vocab_df["is_peace"] = vocab_df.token == " Peace"
# %%
is_word = []
for s in vocab_df.token:
    if len(s) == 0:
        is_word.append(False)
    elif s[0]==" " and s[1:].isalpha() and len(s)>3 and s[2:].islower():
        is_word.append(True)
    else:
        is_word.append(False)
vocab_df["is_word"] = is_word
print(vocab_df["is_word"].sum())
display(vocab_df[vocab_df.is_word].head(50))


# %%
N = vocab_df["is_word"].sum()
word_tokens = vocab_df.query("is_word")["index"].values
PREFIX_LEN = 5
def make_random_prefix(num_prompts, prefix_len=PREFIX_LEN):
    random_index = torch.randint(0, len(word_tokens), (num_prompts, prefix_len))
    random_tokens = torch.tensor(word_tokens)[random_index]
    return random_tokens

peace_tokens = torch.zeros((N, 3+PREFIX_LEN), dtype=torch.int64).cuda()
peace_tokens[:, 0] = model.tokenizer.bos_token_id
peace_tokens[:, 1:1+PREFIX_LEN] = make_random_prefix(N)
peace_tokens[:, -2] = torch.tensor(word_tokens, device="cuda")
peace_tokens[:, -1] = model.to_single_token(" Peace")

_, peace_cache = model.run_with_cache(peace_tokens, return_type=None)
peace_neuron_acts = peace_cache["post", 0][:, -1, :]


nobel_tokens = torch.zeros((N, 3+PREFIX_LEN), dtype=torch.int64).cuda()
nobel_tokens[:, 0] = model.tokenizer.bos_token_id
nobel_tokens[:, 1:1+PREFIX_LEN] = make_random_prefix(N)
nobel_tokens[:, -2] = model.to_single_token(" Nobel")
nobel_tokens[:, -1] = torch.tensor(word_tokens, device="cuda")

_, nobel_cache = model.run_with_cache(nobel_tokens, return_type=None)
nobel_neuron_acts = nobel_cache["post", 0][:, -1, :]

# %%
neuron_df["wout_norm"] = model.W_out[0].norm(dim=1).tolist()
# %%
num_pairs = 2
pairs_tokens = torch.zeros((N*num_pairs, 3+PREFIX_LEN), dtype=torch.int64).cuda()
pairs_tokens[:, 0] = model.tokenizer.bos_token_id
pairs_tokens[:, 1:1+PREFIX_LEN] = make_random_prefix(N * num_pairs)
pairs_tokens[:, -2] = einops.repeat(torch.tensor(word_tokens, device="cuda"), "n -> (m n)", m=num_pairs)
pairs_tokens[:, -1] = torch.cat([
    torch.tensor(word_tokens)[torch.randperm(N)] for i in range(num_pairs)
]).cuda()

_, pairs_cache = model.run_with_cache(pairs_tokens, return_type=None)
pairs_neuron_acts = pairs_cache["post", 0][:, -1, :]
# %%
num_prompts = 5000
clean_tokens = torch.zeros((num_prompts, 3+PREFIX_LEN), dtype=torch.int64).cuda()
clean_tokens[:, 0] = model.tokenizer.bos_token_id
clean_tokens[:, 1:1+PREFIX_LEN] = make_random_prefix(num_prompts)
clean_tokens[:, -2] = model.to_single_token(" Nobel")
clean_tokens[:, -1] = model.to_single_token(" Peace")

_, clean_cache = model.run_with_cache(clean_tokens, return_type=None)
clean_neuron_acts = clean_cache["post", 0][:, -1, :]
# %%
neuron_df["clean_med"] = clean_neuron_acts.median(dim=0).values.tolist()
neuron_df["nobel_med"] = nobel_neuron_acts.median(dim=0).values.tolist()
neuron_df["peace_med"] = peace_neuron_acts.median(dim=0).values.tolist()
neuron_df["pairs_med"] = pairs_neuron_acts.median(dim=0).values.tolist()

# %%
neuron_df["and_term"] = (neuron_df["clean_med"] - neuron_df["pairs_med"]) - (neuron_df["nobel_med"] - neuron_df["pairs_med"]) - (neuron_df["peace_med"] - neuron_df["pairs_med"])
neuron_df["clean_dla"] = neuron_df.clean_med * neuron_df.wdla
neuron_df["clean_v_peace_dla"] = (neuron_df.clean_med - neuron_df.peace_med) * neuron_df.wdla
nutils.show_df(neuron_df[["clean_med", "nobel_med", "peace_med", "pairs_med", "and_term", "wdla", "clean_v_peace_dla"]].sort_values("and_term", ascending=False).head(50))
# %%
def get_prize_lp(tokens):
    logits = model(tokens)[:, -1, :]
    return F.log_softmax(logits, dim=-1)[:, PRIZE]
def get_batched_prize_lp(tokens):
    out = []
    for i in range(0, len(tokens), 1000):
        out.append(get_prize_lp(tokens[i:i+1000]))
    return torch.cat(out)
histogram(get_batched_prize_lp(clean_tokens))
# peace_lps = get_batched_prize_lp(peace_tokens)
# nobel_lps = get_batched_prize_lp(nobel_tokens)
# pairs_lps = get_batched_prize_lp(pairs_tokens)
# temp_df = pd.DataFrame({
#     "prize_lp": torch.cat([peace_lps, nobel_lps, pairs_lps]).tolist(),
#     "label": ["peace"]*N + ["nobel"]*N + ["pairs"]*len(pairs_lps),
# })
# px.histogram(temp_df, x="prize_lp", color="label", barmode="overlay")
# # %%

# W_out_dir = model.W_out[0] @ unembed_dir
# clean_lps = (clean_neuron_acts @ W_out_dir)[None] / cache["scale"][:, -1, 0]
# peace_lps = peace_neuron_acts @ W_out_dir / peace_cache["scale"][:, -1, 0]
# nobel_lps = nobel_neuron_acts @ W_out_dir / nobel_cache["scale"][:, -1, 0]
# pairs_lps = pairs_neuron_acts @ W_out_dir / pairs_cache["scale"][:, -1, 0]
# temp_df = pd.DataFrame({
#     "prize_lp": torch.cat([clean_lps, peace_lps, nobel_lps, pairs_lps]).tolist(),
#     "label": ["nobel peace"]+["peace"]*N + ["nobel"]*N + ["pairs"]*len(pairs_lps),
#     "word": ["nobel peace"]+vocab_df.query("is_word")["unique_token"].tolist()*6,
# })
# px.histogram(temp_df, x="prize_lp", color="label", barmode="overlay", title="Prize DLA from MLP", histnorm="percent").show()
# temp_df.query("label=='peace'").sort_values("prize_lp", ascending=False).head(50)
# %%
x = []
for q in np.linspace(0, 1, 101):
    quantile = neuron_df.and_term.quantile(q)
    x.append(float(neuron_df[neuron_df.and_term>quantile].clean_dla.sum()))
line(x)

# %%
print(((neuron_df.and_term>0.5) & (neuron_df.wdla>0)).sum())
print(((neuron_df.and_term>0.5) & (neuron_df.wdla<0)).sum())
print(((neuron_df.and_term<-0.5) & (neuron_df.wdla>0)).sum())
print(((neuron_df.and_term<-0.5) & (neuron_df.wdla<0)).sum())
# %%
print(neuron_df[neuron_df.and_term>0.5].wdla.mean())
print(neuron_df[neuron_df.and_term<-0.5].wdla.mean())

# %%
neuron_df["and_dla"] = neuron_df["and_term"] * neuron_df["wdla"]
px.histogram(neuron_df, x="and_dla")
# %%
top_neurons = neuron_df[(neuron_df.and_term>0.7) & (neuron_df.and_term < 1.0)].N.values
temp_df = nutils.create_vocab_df(model.W_out[0, top_neurons].mean(0) @ model.W_U, model=model)
temp_df["rank"] = np.arange(len(temp_df))
nutils.show_df(temp_df.head(50))
temp_df.loc[PRIZE]
# %%
