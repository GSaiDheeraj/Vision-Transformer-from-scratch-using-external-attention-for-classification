# Vision-Transformer-from-scratch-using-external-attention-for-classification

paper: https://arxiv.org/pdf/2105.02358.pdf

![image](https://user-images.githubusercontent.com/60067496/203231021-bd8a33b7-980a-43ae-a605-07e77bce369a.png)

### Algorithm
F = query _linear(F) # s h a p e = (B , N, C)

attn = M_k( F ) # s h a p e = (B , N, M)

attn = softmax (attn , dim=1 )

attn = l1_norm ( attn , dim=2 )

out = M_v( attn ) # s h a p e = (B , N, C)

![image](https://user-images.githubusercontent.com/60067496/203231822-c9701f22-40d8-495b-bc0d-9e448afce8f8.png)

external attention, which computes attention between the
input pixels and an external memory unit M ∈ R^(S×d), via:
A = (α)i,j = Norm(FM)

Fout = AM.

A is the attention map inferred from this learned dataset-level prior
knowledge; Finally, we update the input features from M by the similarities in A

The computational complexity of external attention is
O(dSN); as d and S are hyper-parameters
