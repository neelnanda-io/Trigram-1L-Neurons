# %%
from neel.imports import *
from neel_plotly import *
model = HookedTransformer.from_pretrained("gelu-1l")
utils.test_prompt(" Nobel Peace", "Prize", model)
torch.set_grad_enabled(False)
# %%
utils.test_prompt(" Noble Peace", "Prize", model)
utils.test_prompt(" Nobel peas", "Prize", model)
# %%
NOBEL = model.to_single_token(" Nobel")
# %%
logits, cache = model.run_with_cache(" Nobel Peace")
unembed_dir = model.tokens_to_residual_directions(" Peace")
resid_stack, labels = cache.get_full_resid_decomposition(apply_ln=True, pos_slice=-1, return_labels=True)

line(resid_stack @ unembed_dir, x=labels, title="DLA of components on clean input")
# %%
corr_logits, corr_cache = model.run_with_cache(" Noble Peace")
corr_resid_stack, corr_labels = corr_cache.get_full_resid_decomposition(apply_ln=True, pos_slice=-1, return_labels=True)

line([(resid_stack @ unembed_dir), (corr_resid_stack @ unembed_dir), ((resid_stack-corr_resid_stack) @ unembed_dir)], x=labels, title="DLA of components on clean input", line_labels=["clean", "corr", "diff"])
# %%
INVESTIGATE_NEURON_424 = False
if INVESTIGATE_NEURON_424:
    all_vocab_tokens = torch.zeros((model.cfg.d_vocab, 2), dtype=torch.int64).cuda()
    all_vocab_tokens[:, 0] = model.tokenizer.bos_token_id
    all_vocab_tokens[:, 1] = torch.arange(model.cfg.d_vocab).cuda()

    _, all_vocab_cache = model.run_with_cache(all_vocab_tokens, return_type=None)
    full_vocab = model.to_str_tokens(torch.arange(model.cfg.d_vocab))
    line(all_vocab_cache["post", 0][:, -1, 424], x=[f"{s}/{i}" for i, s in enumerate(full_vocab)], title="Neuron 424 act per token")
    
    vocab_df = pd.DataFrame({
        "token": full_vocab,
        "unique_token": [f"{s}/{i}" for i, s in enumerate(full_vocab)],
        "index": torch.arange(model.cfg.d_vocab).tolist(),
        "neuron_424": all_vocab_cache["post", 0][:, -1, 424].tolist(),
    })
    px.line(vocab_df.sort_values("neuron_424", ascending=False), x="unique_token", y="neuron_424", title="Neuron 424 act per token")
    
    vocab_df["has_space"] = vocab_df.token.str.contains(" ")
    vocab_df["start_cap"] = [len(s.strip())>0 and s.strip()[0].isupper() for s in vocab_df.token]
    vocab_df["is_capital"] = vocab_df["has_space"] & vocab_df["start_cap"]

    px.histogram(vocab_df, color="has_space", x="neuron_424", marginal="box", barmode="overlay", hover_name="token").show()
    px.histogram(vocab_df, color="start_cap", x="neuron_424", marginal="box", barmode="overlay", hover_name="token").show()
    px.histogram(vocab_df, color="is_capital", x="neuron_424", marginal="box", barmode="overlay", hover_name="token").show()
# %%
# Fix the headline neuron to its clean value
CLEAN_424_VALUE = 0.305
def fix_n424(mlp_post, hook):
    mlp_post[:, :, 424] = CLEAN_424_VALUE
    return mlp_post
# model.add_perma_hook(utils.get_act_name("post", 0, "mlp"), fix_n424)
# %%
clean_dla = (resid_stack @ unembed_dir).squeeze(-1)
corr_dla = (corr_resid_stack @ unembed_dir).squeeze(-1)
diff_dla = clean_dla - corr_dla
# line([, (corr_resid_stack @ unembed_dir), ((resid_stack-corr_resid_stack) @ unembed_dir)], x=labels, title="DLA of components on clean input", line_labels=["clean", "corr", "diff"])
scatter(x=diff_dla, y=clean_dla, color=corr_dla, hover=labels, xaxis="Diff", yaxis="Clean", title="DLA of components on clean vs diff")

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
corr_prompt = " Noble Peace"
corr_logits, corr_cache = model.run_with_cache(corr_prompt)
corr_resid_stack, corr_labels = corr_cache.get_full_resid_decomposition(apply_ln=True, pos_slice=-1, return_labels=True)

line([(resid_stack @ unembed_dir), (corr_resid_stack @ unembed_dir), ((resid_stack-corr_resid_stack) @ unembed_dir)], x=labels, title=f"DLA of components on clean input vs {corr_prompt}", line_labels=["clean", "corr", "diff"],)
clean_dla = (resid_stack @ unembed_dir).squeeze(-1)
corr_dla = (corr_resid_stack @ unembed_dir).squeeze(-1)
diff_dla = clean_dla - corr_dla
# line([, (corr_resid_stack @ unembed_dir), ((resid_stack-corr_resid_stack) @ unembed_dir)], x=labels, title="DLA of components on clean input", line_labels=["clean", "corr", "diff"])
scatter(x=diff_dla, y=clean_dla, color=corr_dla, hover=labels, xaxis="Diff", yaxis="Clean", title=f"DLA of components on clean vs diff on {corr_prompt}")
scatter(y=clean_dla, x=corr_dla, hover=labels, xaxis="Corr", yaxis="Clean", title=f"DLA of components on clean vs diff on {corr_prompt}")

neuron_wdla = model.W_out[0] @ model.W_U[:, model.to_single_token(" Prize")]
scatter(y=cache["post", 0][0, -1], x=corr_cache["post", 0][0, -1], hover=np.arange(model.cfg.d_mlp), xaxis="Corr", yaxis="Clean", title=f"Neuron acts on clean vs {corr_prompt}", color=neuron_wdla)


# %%
for token in [" John", "John", " Prize", " peas"]:
    px.histogram(to_numpy(model.W_out[0] @ model.W_U[:, model.to_single_token(token)]), hover_name=list(map(str, np.arange(model.cfg.d_mlp))), marginal="box", title=token).show()
# %%
no_prefix_logits, no_prefix_cache = model.run_with_cache(" Peace")
no_prefix_neuron_acts = no_prefix_cache["post", 0][0, -1]
# %%
neuron_df = nutils.make_neuron_df(model.cfg.n_layers, model.cfg.d_mlp)
clean_neuron_acts = cache["post", 0][0, -1]
neuron_df["clean"] = clean_neuron_acts.tolist()
corr_neuron_acts = corr_cache["post", 0][0, -1]
neuron_df["corr"] = corr_neuron_acts.tolist()
neuron_df["no_prefix"] = no_prefix_neuron_acts.tolist()
neuron_df["diff"] = neuron_df["clean"] - neuron_df["corr"]

PRIZE = model.to_single_token(" Prize")
neuron_df["wdla"] = (model.W_out[0] @ model.W_U[:, PRIZE]).tolist()
neuron_df["clean_dla"] = (neuron_df["clean"] * neuron_df["wdla"]).tolist()
neuron_df["corr_dla"] = (neuron_df["corr"] * neuron_df["wdla"]).tolist()
neuron_df["no_prefix_dla"] = (neuron_df["no_prefix"] * neuron_df["wdla"]).tolist()
neuron_df["diff_dla"] = (neuron_df["clean_dla"] - neuron_df["corr_dla"]).tolist()

nutils.show_df(neuron_df.sort_values("diff_dla", ascending=False).head(30))
nutils.show_df(neuron_df.sort_values("diff", ascending=False).head(30))

# %%
top_k = 20
top_neurons_by_act = neuron_df.sort_values("diff_dla", ascending=False).head(top_k).N.values

temp_df = nutils.create_vocab_df((model.W_out[0, top_neurons_by_act]).sum(0) @ model.W_U, model=model)
# temp_df = nutils.create_vocab_df((model.W_out[0, top_neurons_by_act] * clean_neuron_acts[top_neurons_by_act, None]).sum(0) @ model.W_U, model=model)
temp_df["rank"] = np.arange(len(temp_df))
print(temp_df.loc[PRIZE])
nutils.show_df(temp_df.head(50))
# %%

# Investigate 939
INVESTIGATE_939 = False
if INVESTIGATE_939:
    n939_wdla = model.W_out[0, 939] @ model.W_U
    n939_df = copy.deepcopy(vocab_df)
    n939_df["n939_wdla"] = n939_wdla.tolist()
    px.histogram(n939_df, color="has_space", x="n939_wdla", marginal="box", barmode="overlay", hover_name="token").show()
    px.histogram(n939_df, color="start_cap", x="n939_wdla", marginal="box", barmode="overlay", hover_name="token").show()
    px.histogram(n939_df, color="is_capital", x="n939_wdla", marginal="box", barmode="overlay", hover_name="token").show()
    n939_df["space_case"] = n939_df["has_space"] * 2 + n939_df["start_cap"]
    px.histogram(n939_df, color="space_case", x="n939_wdla", marginal="box", barmode="overlay", hover_name="token").show()

# %%
# prefix_tokens_temp = [
#     " John",
#     "John",
#     " Alice",
#     "Alice",
#     " peas",
#     " table",
#     " war",
#     " carrot",
#     " Fields",
#     " Noble",
#     " Falcon",
#     " Turing",
#     " Johnson",
#     " Wes",
#     " Lamp",
#     "Light"
# ]
# prefix_tokens = [i for i in prefix_tokens_temp if len(model.to_str_tokens(i, prepend_bos=False))==1]
# print(set(prefix_tokens_temp) - set(prefix_tokens))
# prompts = [f"{p} Peace" for p in prefix_tokens]
# tokens = model.to_tokens(prompts)

N = model.cfg.d_vocab
all_vocab_tokens_peace = torch.zeros((N, 3), dtype=torch.int64).cuda()
all_vocab_tokens_peace[:, 0] = model.tokenizer.bos_token_id
all_vocab_tokens_peace[:, 1] = torch.arange(N).cuda()
all_vocab_tokens_peace[:, 2] = model.to_single_token(" Peace")

_, all_vocab_cache = model.run_with_cache(all_vocab_tokens_peace, return_type=None)
# all_vocab_logits = all_vocab_logits[:, -1, :]
# all_vocab_log_probs = all_vocab_logits.log_softmax(dim=-1)[:, PRIZE]

full_vocab = model.to_str_tokens(np.arange(N))
# line(all_vocab_cache["post", 0][:, -1, 424], x=[f"{s}/{i}" for i, s in enumerate(full_vocab)], title="Neuron 424 act per token")

vocab_df = pd.DataFrame({
    "token": full_vocab,
    "unique_token": nutils.process_tokens_index(np.arange(N), model),
    "index": torch.arange(N).tolist(),
    # "prize_log_probs": all_vocab_log_probs.tolist(),
})
# px.line(vocab_df.sort_values("prize_log_probs", ascending=False), x="unique_token", y="prize_log_probs", title="Prize log prob per token").show()

vocab_df["has_space"] = vocab_df.token.str.contains(" ")
vocab_df["start_cap"] = [len(s.strip())>0 and s.strip()[0].isupper() for s in vocab_df.token]
vocab_df["is_capital"] = vocab_df["has_space"] & vocab_df["start_cap"]
vocab_df["is_nobel"] = vocab_df.token == " Nobel"

# px.histogram(vocab_df, color="has_space", x="prize_log_probs", marginal="box", barmode="overlay", hover_name="token").show()
# px.histogram(vocab_df, color="start_cap", x="prize_log_probs", marginal="box", barmode="overlay", hover_name="token").show()
# px.histogram(vocab_df, color="is_capital", x="prize_log_probs", marginal="box", barmode="overlay", hover_name="token").show()
# %%
all_vocab_neuron_acts = all_vocab_cache["post", 0][:, -1, :]
all_vocab_neuron_acts.shape
px.box(vocab_df, y=all_vocab_neuron_acts[:, 939].tolist(), x="has_space")

# %%
vocab_df["is_alpha"] = [t.strip().isalpha() for t in vocab_df.token]
# %%

vocab_df["939"] = all_vocab_cache["pre", 0][:, -1, 939].tolist()

nutils.show_df(vocab_df[vocab_df["is_alpha"]].sort_values("939").head(30))

# %%
neuron_df = neuron_df.drop(columns=["corr", "no_prefix", "corr_dla", "no_prefix_dla"])
# %%
all_vocab_dla = all_vocab_neuron_acts * torch.tensor(neuron_df["wdla"].values[None, :]).cuda()
neuron_df["median"] = all_vocab_neuron_acts.median(0).values.tolist()
neuron_df["median_diff"] = neuron_df["clean"] - neuron_df["median"]
neuron_df["median_dla"] = all_vocab_dla.median(0).values.tolist()
neuron_df["median_diff_dla"] = neuron_df["clean_dla"] - neuron_df["median_dla"]

nutils.show_df(neuron_df.sort_values("median_diff_dla", ascending=False).head(30))
nutils.show_df(neuron_df.sort_values("median_diff_dla", ascending=False).tail(30))
# %%
neuron_df["inhibited"] = neuron_df["median"] > neuron_df["clean"]

px.box(neuron_df, x="inhibited", y="median_diff_dla", hover_name="N", title="Inhibited vs Excited neurons")
# %%
neuron_df.groupby("inhibited")["median_diff_dla"].sum() / neuron_df["median_diff_dla"].sum()
# %%
upper = []
lower = []

for thresh in np.linspace(-0.2, 0.3, 100):
    filtered = neuron_df["median_diff_dla"]>thresh
    x = neuron_df[filtered].groupby("inhibited")["median_diff_dla"].sum() / neuron_df[filtered]["median_diff_dla"].sum()
    try:
        upper.append(float(x[False]))
    except:
        upper.append(0.)
    try:
        lower.append(float(x[True]))
    except:
        lower.append(0.)
line([upper, lower], line_labels=["excited", "inhibited"], title="Excited vs Inhibited neurons", x=np.linspace(-0.2, 0.3, 100))
# %%
px.histogram(neuron_df, "median_diff_dla", marginal="box", hover_name="N", color="inhibited", barmode="overlay", title="DLA relative to median by inhibited vs excited")
# %%
is_prev_nobel = cache["resid_mid", 0][0, -1, :] - all_vocab_cache["resid_mid", 0][vocab_df["is_alpha"].values, -1, :].mean(0)
# %%
all_vocab_tokens_nobel = torch.zeros((N, 3), dtype=torch.int64).cuda()
all_vocab_tokens_nobel[:, 0] = model.tokenizer.bos_token_id
all_vocab_tokens_nobel[:, 1] = model.to_single_token(" Nobel")
all_vocab_tokens_nobel[:, 2] = torch.arange(N).cuda()

_, all_vocab_nobel_cache = model.run_with_cache(all_vocab_tokens_nobel, return_type=None)

is_curr_peace = cache["resid_mid", 0][0, -1, :] - all_vocab_nobel_cache["resid_mid", 0][vocab_df["is_alpha"].values, -1, :].mean(0)

# %%
neuron_df["peace"] = (is_curr_peace @ model.W_in[0]).tolist()
neuron_df["nobel"] = (is_prev_nobel @ model.W_in[0]).tolist()
neuron_df["inject"] = neuron_df["peace"] + neuron_df["nobel"]

nutils.show_df(neuron_df[["N", "clean", "median", "wdla", "median_diff_dla", "peace", "nobel", "inject"]].sort_values("median_diff_dla", ascending=False).head(30))

scatter(x=neuron_df.nobel, y=neuron_df.median_diff_dla, color=neuron_df.wdla, xaxis="nobel", yaxis="Diff DLA", hover=neuron_df.N, title="nobel vs Diff DLA (color=wDLA)")
scatter(x=neuron_df.nobel, y=neuron_df.median_diff_dla, color=neuron_df.inhibited, xaxis="nobel", yaxis="Diff DLA", hover=neuron_df.N, title="nobel vs Diff DLA (color=inhibited)")

# %%
