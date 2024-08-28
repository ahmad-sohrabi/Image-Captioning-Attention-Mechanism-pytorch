import torch
from torch import nn
import torchvision
from torchvision.models import ResNet18_Weights
from models.torch.layers import embedding_layer


class Encoder(nn.Module):

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune(fine_tune=True)

    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=False):
        for p in self.resnet.parameters():
            p.requires_grad = True
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5,
                 embedding_matrix=None, train_embd=True):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = embedding_layer(num_embeddings=vocab_size, embedding_dim=embed_dim,
                                         embedding_matrix=embedding_matrix, trainable=train_embd)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)

        h, c = self.init_hidden_state(encoder_out)


        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar,
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def sample(self, encoder_out, startseq_idx, endseq_idx=-1, max_len=40, return_alpha=False):

        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        batch_size = encoder_out.size(0)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        h, c = self.init_hidden_state(encoder_out)

        sampled_ids = []
        alphas = []

        prev_timestamp_words = torch.LongTensor([[startseq_idx]] * batch_size).to(encoder_out.device)
        for i in range(max_len):
            embeddings = self.embedding(prev_timestamp_words).squeeze(1)
            awe, alpha = self.attention(encoder_out, h)
            alpha = alpha.view(-1, enc_image_size, enc_image_size).unsqueeze(1)

            gate = self.sigmoid(self.f_beta(h))  # gating scalar
            awe = gate * awe

            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            predicted_prob = self.fc(h)
            predicted = predicted_prob.argmax(1)

            sampled_ids.append(predicted)
            alphas.append(alpha)

            prev_timestamp_words = predicted.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return (sampled_ids, torch.cat(alphas, 1)) if return_alpha else sampled_ids


class Captioner(nn.Module):
    def __init__(self, encoded_image_size, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512,
                 dropout=0.5, **kwargs):
        super().__init__()
        self.encoder = Encoder(encoded_image_size=encoded_image_size)
        self.decoder = DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size,
                                            encoder_dim, dropout)

    def forward(self, images, encoded_captions, caption_lengths):

        encoder_out = self.encoder(images)
        decoder_out = self.decoder(encoder_out, encoded_captions, caption_lengths.unsqueeze(1))
        return decoder_out

    def sample(self, images, startseq_idx, endseq_idx=-1, max_len=40, return_alpha=False):
        encoder_out = self.encoder(images)
        return self.decoder.sample(encoder_out=encoder_out, startseq_idx=startseq_idx, max_len=max_len,
                                   return_alpha=return_alpha)
