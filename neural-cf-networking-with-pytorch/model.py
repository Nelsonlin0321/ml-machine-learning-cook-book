
import torch

class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config

        # matrix factorization part
        self.embedding_user_mf = torch.nn.Embedding(
            num_embeddings=self.config.num_user, embedding_dim=self.config.latent_dim_mf)
        if config.use_xavier_uniform:
            torch.nn.init.xavier_uniform_(self.embedding_user_mf.weight)

        self.embedding_item_mf = torch.nn.Embedding(
            num_embeddings=self.config.num_item, embedding_dim=self.config.latent_dim_mf)
        if config.use_xavier_uniform:
            torch.nn.init.xavier_uniform_(self.embedding_item_mf.weight)

        # multilayer perceptron part
        self.embedding_user_mlp = torch.nn.Embedding(
            num_embeddings=self.config.num_user, embedding_dim=self.config.latent_dim_mlp)
        if config.use_xavier_uniform:
            torch.nn.init.xavier_uniform_(self.embedding_user_mlp.weight)

        self.embedding_item_mlp = torch.nn.Embedding(
            num_embeddings=self.config.num_item, embedding_dim=self.config.latent_dim_mlp)
        
        if config.use_xavier_uniform:
            torch.nn.init.xavier_uniform_(self.embedding_item_mlp.weight)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.config.layers[:-1], self.config.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.logits = torch.nn.Linear(
            in_features=self.config.layers[-1] + self.config.latent_dim_mf, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # mf part: element-wise product
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        mf_vector = torch.nn.Dropout(self.config.dropout_rate_mf)(mf_vector)

        # mlp part
        # the concat latent vector
        mlp_vector = torch.cat(
            [user_embedding_mlp, item_embedding_mlp], dim=-1)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            """
            1) The sigmoid function restricts each
            neuron to be in (0,1), which may limit the model's perfor-
            mance; and it is known to suffer from saturation, where
            neurons stop learning when their output is near either 0 or
            1. 2) Even though tanh is a better choice and has been
            widely adopted [6, 44], it only alleviates the issues of sig-
            moid to a certain extent, since it can be seen as a rescaled
            version of sigmoid. And 3) as
            such, we opt for ReLU, which is more biologically plausi-
            ble and proven to be non-saturated [9]; moreover, it encour-
            ages sparse activations, being well-suited for sparse data and
            making the model less likely to be overfitting. Our empirical
            results show that ReLU yields slightly better performance
            than tanh, which in turn is significantly better than sigmoid.
            """
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        mlp_vector = torch.nn.Dropout(self.config.dropout_rate_mlp)(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.logits(vector)
        output = self.sigmoid(logits)
        return output
