from hyrag.embeddings import MistralEmbedder, cosine_similarity

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials.",
    "Passwords rotate every 180 days.",
    "No deploys on Fridays.",
    "Production releases are blocked at the end of the week.",
    "ERR_TUNNEL_4012",
]

embedder = MistralEmbedder()
vectors = embedder.embed(sentences)

print(f"Each sentence became a vector of {vectors.shape[1]} numbers.")
print(f"First 8 numbers of sentence 0: {vectors[0][:8].round(3)}\n")

print("Cosine similarity between every pair (1.0 = same meaning):\n")
print(" " * 5 + "".join(f"{j:>7}" for j in range(len(sentences))))
for i in range(len(sentences)):
    row = "".join(f"{cosine_similarity(vectors[i], vectors[j]):7.2f}" for j in range(len(sentences)))
    print(f"{i:>4} {row}   {sentences[i]}")
