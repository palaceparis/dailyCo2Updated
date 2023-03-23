def forward(self, data, device):
    """
    Forward pass of the ST-GAT model
    :param data Data to make a pass on
    :param device Device to operate on
    """
    x, edge_index_1, edge_attr_1, edge_index_2, edge_attr_2 = (
        data.x,
        data.edge_index[0:140],
        data.edge_attr[0:140],
        data.edge_index[141:465],
        data.edge_attr[141:465],
    )
    # apply dropout
    if device == "cpu":
        x = torch.FloatTensor(x)
    else:
        x = torch.cuda.FloatTensor(x)

        # gat layer: output of gat: [11400, 12]
        x = self.gat(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)

        # RNN: 2 LSTM
        # [batchsize*n_nodes, seq_length] -> [batch_size, n_nodes, seq_length]
        batch_size = data.num_graphs
        n_node = int(data.num_nodes / batch_size)
        x = torch.reshape(x, (batch_size, n_node, data.num_features))
        # for lstm: x should be (seq_length, batch_size, n_nodes)
        # sequence length = 12, batch_size = 50, n_node = 228
        x = torch.movedim(x, 2, 0)
        # [12, 50, 228] -> [12, 50, 32]
        x, _ = self.lstm1(x)
        # [12, 50, 32] -> [12, 50, 128]
        x, _ = self.lstm2(x)

        # Output contains h_t for each timestep, only the last one has all input's accounted for
        # [12, 50, 128] -> [50, 128]
        x = torch.squeeze(x[-1, :, :])
        # [50, 128] -> [50, 228*9]
        x = self.linear(x)
        # Now reshape into final output
        s = x.shape
        # [50, 228*9] -> [50, 228, 9]
        x = torch.reshape(x, (s[0], self.n_nodes, self.n_pred))

        # [50, 228, 9] ->  [11400, 9]
        x = torch.reshape(x, (s[0] * self.n_nodes, self.n_pred))
        return x
