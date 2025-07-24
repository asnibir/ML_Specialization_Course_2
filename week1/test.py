import gensim.downloader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import numpy as np

# Load the GloVe model
model = gensim.downloader.load("glove-wiki-gigaword-50")

# List of words to visualize
words = ["tower", "building", "castle", "bridge", "sky", "cloud", "tree", "city", "house", "apartment"]

print(model["tower"])

# Get vectors and convert to numpy array
word_vectors = np.array([model[word] for word in words])

# Reduce to 3D using t-SNE
tsne = TSNE(n_components=3, random_state=0, perplexity=5)
vectors_3d = tsne.fit_transform(word_vectors)

# Plot in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for i, word in enumerate(words):
    x, y, z = vectors_3d[i]
    ax.scatter(x, y, z)
    ax.text(x, y, z, word, fontsize=10)

ax.set_title("3D t-SNE Visualization of Word Vectors")
plt.show()

# import gensim.downloader
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import numpy as np

# # Load GloVe vectors (50D)
# model = gensim.downloader.load("glove-wiki-gigaword-50")

# # Words to visualize
# words = ["the", "big", "idea", "as", "a", "model", "tweaks", "and", "tunes", "its", "weights"]

# # Get word vectors
# vectors = np.array([model[word] for word in words])

# # Reduce to 2D using PCA for clearer directional structure
# pca = PCA(n_components=2)
# vectors_2d = pca.fit_transform(vectors)

# # Plot
# plt.figure(figsize=(10, 10), facecolor='black')
# ax = plt.gca()
# ax.set_facecolor("black")

# for i, word in enumerate(words):
#     x, y = vectors_2d[i]
#     # Draw arrow from origin to (x, y)
#     plt.arrow(0, 0, x, y, head_width=0.1, head_length=0.1, fc='skyblue', ec='skyblue', length_includes_head=True)
#     # Label the word
#     plt.text(x * 1.05, y * 1.05, word, fontsize=12, color='skyblue', ha='center', va='center')

# # Axes styling
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
# plt.grid(False)
# plt.axis('off')
# plt.title("2D Word Vectors from GloVe", color='white', fontsize=14)
# plt.tight_layout()
# plt.show()

