def generate_and_print_sample(model, tokeniser, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokeniser).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model = model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokeniser)
    print(decoded_text.replace("\n", " "))
    model.train()