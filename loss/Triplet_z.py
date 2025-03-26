import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(x, y, eps=1e-12):
	"""
	Args:
	  x: pytorch Tensor, with shape [m, d]
	  y: pytorch Tensor, with shape [n, d]
	Returns:
	  dist: pytorch Tensor, with shape [m, n]
	"""
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(x, y.t(), beta=1, alpha=-2) #dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=eps).sqrt()

	return dist

# def hard_example_mining(dist_mat, target):
# 	"""For each anchor, find the hardest positive and negative sample.
# 	Args:
# 	  dist_mat: pytorch Tensor, pair wise distance between samples, shape [N, N]
# 	  target: pytorch LongTensor, with shape [N]
# 	  return_inds: whether to return the indices. Save time if `False`(?)
# 	Returns:
# 	  dist_ap: pytorch Tensor, distance(anchor, positive); shape [N]
# 	  dist_an: pytorch Tensor, distance(anchor, negative); shape [N]
# 	  p_inds: pytorch LongTensor, with shape [N];
# 	    indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
# 	  n_inds: pytorch LongTensor, with shape [N];
# 	    indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
# 	NOTE: Only consider the case in which all target have same num of samples,
# 	  thus we can cope with all anchors in parallel.
# 	"""
# 	assert len(dist_mat.size()) == 2
# 	assert dist_mat.size(0) == dist_mat.size(1)
# 	N = dist_mat.size(0)
#
# 	# shape [N, N]
# 	is_pos = target.expand(N, N).eq(target.expand(N, N).t())
# 	is_neg = target.expand(N, N).ne(target.expand(N, N).t())
#
# 	dist_ap1, relative_p_inds = torch.max(
# 		dist_mat[is_pos].contiguous().view(N, -1)[:,:8], 1, keepdim=True)
# 	dist_ap2, relative_p_inds = torch.max(
# 		dist_mat[is_pos].contiguous().view(N, -1)[:], 1, keepdim=True)
# 	dist_ap = torch.cat((dist_ap1,dist_ap2),dim=1)
# 	dist_ap = torch.mean(dist_ap,dim=1)
# 	dist_an, relative_n_inds = torch.min(
# 		dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
#
# 	# dist_ap = dist_ap.squeeze(1)
# 	dist_an = dist_an.squeeze(1)
#
# 	return dist_ap, dist_an
def hard_example_mining(dist_mat, target):
	"""For each anchor, find the hardest positive and negative sample.
	Args:
	  dist_mat: pytorch Tensor, pair wise distance between samples, shape [N, N]
	  target: pytorch LongTensor, with shape [N]
	  return_inds: whether to return the indices. Save time if `False`(?)
	Returns:
	  dist_ap: pytorch Tensor, distance(anchor, positive); shape [N]
	  dist_an: pytorch Tensor, distance(anchor, negative); shape [N]
	  p_inds: pytorch LongTensor, with shape [N];
	    indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
	  n_inds: pytorch LongTensor, with shape [N];
	    indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
	NOTE: Only consider the case in which all target have same num of samples,
	  thus we can cope with all anchors in parallel.
	"""
	assert len(dist_mat.size()) == 2
	assert dist_mat.size(0) == dist_mat.size(1)
	N = dist_mat.size(0)

	# shape [N, N]
	is_pos = target.expand(N, N).eq(target.expand(N, N).t())
	is_neg = target.expand(N, N).ne(target.expand(N, N).t())

	dist_ap, relative_p_inds = torch.max(
		dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)

	dist_an, relative_n_inds = torch.min(
		dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

	dist_ap = dist_ap.squeeze(1)
	dist_an = dist_an.squeeze(1)

	return dist_ap, dist_an
class TripletLoss_Balanced_2branch(nn.Module):
	def __init__(self, margin, feat_norm='yes'):
		super(TripletLoss_Balanced_2branch, self).__init__()
		self.margin = margin
		self.feat_norm = feat_norm
		if margin >= 0:
			self.ranking_loss = nn.MarginRankingLoss(margin=margin)
		else:
			self.ranking_loss = nn.SoftMarginLoss()

	def forward(self, global_feat1, global_feat2, target):

		feat_rgb1, feat_ir1 = global_feat1.chunk(2, 0)
		feat_rgb2, feat_ir2 = global_feat2.chunk(2, 0)
		feat_rgb = torch.cat((feat_rgb1,feat_rgb2))
		feat_ir = torch.cat((feat_ir1,feat_ir2))

		dist_mat = euclidean_dist(global_feat1, global_feat2)
		dist_ap, dist_an = hard_example_mining(dist_mat, target)

		N = dist_mat.size(0)
		# shape [N, N]

		is_posf = target.expand(N, N).eq(target.expand(N, N).t()).float()
		is_neg = target.expand(N, N).ne(target.expand(N, N).t())

		dist_mat_an = dist_mat[is_neg].view(128, -1)

		for i in range(128):
			idx_an = torch.nonzero(is_posf[i] < 0.5)
			feat_an1 = global_feat1[idx_an].view(112, -1)[:56]  # 56, 768
			feat_an2 = global_feat1[idx_an].view(112, -1)[56:]  # 56, 768

			# idx_ap = torch.nonzero(is_posf[i] > 0.5)  # ap

			_, idx1 = torch.topk(dist_mat_an[i][:56].view(1, -1), 1, dim=1, largest=False)
			_, idx2 = torch.topk(dist_mat_an[i][56:].view(1, -1), 1, dim=1, largest=False)
			feat_an1 = feat_an1[idx1.detach().cpu().numpy()]
			feat_an2 = feat_an2[idx2.detach().cpu().numpy()]
			feat_an = torch.cat((feat_an1, feat_an2))
			feat_mean_1 = torch.mean(feat_an, dim=0).view(1, -1)

			if i == 0:
				feat_mean = feat_mean_1
			else:
				feat_mean = torch.cat([feat_mean, feat_mean_1], dim=0)

		feat_rgb_mid1, feat_ir_mid1,feat_rgb_mid2, feat_ir_mid2 = feat_mean.chunk(4, 0)
		feat_rgb_mid, feat_ir_mid = torch.cat((feat_rgb_mid1, feat_rgb_mid2)),torch.cat((feat_ir_mid1,feat_ir_mid2))
		dist_an_rgb_mid = torch.diag(euclidean_dist(feat_rgb, feat_rgb_mid))  # [32, 768]-> 32
		dist_an_ir_mid = torch.diag(euclidean_dist(feat_ir, feat_ir_mid))  # [32, 768]

		dist_an = torch.cat([dist_an_rgb_mid, dist_an_ir_mid], dim=0)
		triplets = dist_ap - dist_an + self.margin

		loss = triplets[triplets > 0].sum() / 128
		# print(triplets[triplets > 0].shape[0])
		return loss  , triplets[triplets>0].shape[0]

# branch1和branch2各取一个负样本
class TripletLoss_Balanced_2branch1(nn.Module):
	def __init__(self, margin, feat_norm='yes'):
		super(TripletLoss_Balanced_2branch1, self).__init__()
		self.margin = margin
		self.feat_norm = feat_norm
		if margin >= 0:
			self.ranking_loss = nn.MarginRankingLoss(margin=margin)
		else:
			self.ranking_loss = nn.SoftMarginLoss()

	def forward(self, global_feat1, global_feat2, target):

		feat_rgb1, feat_ir1 = global_feat1.chunk(2, 0)
		feat_rgb2, feat_ir2 = global_feat2.chunk(2, 0)
		feat_rgb = torch.cat((feat_rgb1,feat_rgb2))
		feat_ir = torch.cat((feat_ir1,feat_ir2))

		dist_mat = euclidean_dist(global_feat1, global_feat2)
		dist_ap, dist_an = hard_example_mining(dist_mat, target)

		N = dist_mat.size(0)
		# shape [N, N]

		is_posf = target.expand(N, N).eq(target.expand(N, N).t()).float()
		is_neg = target.expand(N, N).ne(target.expand(N, N).t())

		dist_mat_an = dist_mat[is_neg].view(128, -1)

		for i in range(128):
			idx_an = torch.nonzero(is_posf[i] < 0.5)
			feat_an1 = global_feat1[idx_an].view(112, -1)[:56].detach()  # 56, 768
			feat_an2 = global_feat1[idx_an].view(112, -1)[56:]  # 56, 768

			# idx_ap = torch.nonzero(is_posf[i] > 0.5)  # ap

			_, idx1 = torch.topk(dist_mat_an[i][:56].view(1, -1), 1, dim=1, largest=False)
			_, idx2 = torch.topk(dist_mat_an[i][56:].view(1, -1), 1, dim=1, largest=False)
			feat_an1 = feat_an1[idx1.detach().cpu().numpy()]
			feat_an2 = feat_an2[idx2.detach().cpu().numpy()]
			feat_an = torch.cat((feat_an1, feat_an2))
			feat_mean_1 = torch.mean(feat_an, dim=0).view(1, -1)

			if i == 0:
				feat_mean = feat_mean_1
			else:
				feat_mean = torch.cat([feat_mean, feat_mean_1], dim=0)

		feat_rgb_mid1, feat_ir_mid1,feat_rgb_mid2, feat_ir_mid2 = feat_mean.chunk(4, 0)
		feat_rgb_mid, feat_ir_mid = torch.cat((feat_rgb_mid1, feat_rgb_mid2)),torch.cat((feat_ir_mid1,feat_ir_mid2))
		dist_an_rgb_mid = torch.diag(euclidean_dist(feat_rgb, feat_rgb_mid))  # [32, 768]-> 32
		dist_an_ir_mid = torch.diag(euclidean_dist(feat_ir, feat_ir_mid))  # [32, 768]

		dist_an = torch.cat([dist_an_rgb_mid, dist_an_ir_mid], dim=0)
		triplets = dist_ap - dist_an + self.margin

		loss = triplets[triplets > 0].sum() / 128
		# print(triplets[triplets > 0].shape[0])
		return loss  , triplets[triplets>0].shape[0]

class TripletLoss_Balanced(nn.Module):
	def __init__(self, margin, feat_norm='yes'):
		super(TripletLoss_Balanced, self).__init__()
		self.margin = margin
		self.feat_norm = feat_norm
		if margin >= 0:
			self.ranking_loss = nn.MarginRankingLoss(margin=margin)
		else:
			self.ranking_loss = nn.SoftMarginLoss()

	def forward(self, global_feat1, global_feat2, target):

		feat_rgb1, feat_ir1 = global_feat1.chunk(2, 0)
		feat_rgb2, feat_ir2 = global_feat2.chunk(2, 0)
		feat_rgb = torch.cat((feat_rgb1, feat_rgb2))
		feat_ir = torch.cat((feat_ir1, feat_ir2))

		dist_mat = euclidean_dist(global_feat1, global_feat2)
		dist_ap, dist_an = hard_example_mining(dist_mat, target)

		N = dist_mat.size(0)
		# shape [N, N]

		is_posf = target.expand(N, N).eq(target.expand(N, N).t()).float()
		is_neg = target.expand(N, N).ne(target.expand(N, N).t())

		dist_mat_an = dist_mat[is_neg].view(128, -1)

		for i in range(128):
			idx_an = torch.nonzero(is_posf[i] < 0.5)
			feat_an = global_feat1[idx_an].view(112,-1) # 56, 768
			feat_an = torch.cat((feat_an[:56].detach(),feat_an[56:]))

			#idx_ap = torch.nonzero(is_posf[i] > 0.5)  # ap

			_, idx = torch.topk(dist_mat_an[i].view(1, -1), 2, dim=1, largest=False)
			feat_an1 = feat_an[idx.detach().cpu().numpy()]
			feat_mean_1 = torch.mean(feat_an1, dim=0).view(1, -1)

			if i == 0:
				feat_mean = feat_mean_1
			else:
				feat_mean = torch.cat([feat_mean,feat_mean_1],dim=0)

		feat_rgb_mid1, feat_ir_mid1, feat_rgb_mid2, feat_ir_mid2 = feat_mean.chunk(4, 0)
		feat_rgb_mid, feat_ir_mid = torch.cat((feat_rgb_mid1, feat_rgb_mid2)), torch.cat((feat_ir_mid1, feat_ir_mid2))
		dist_an_rgb_mid = torch.diag(euclidean_dist(feat_rgb, feat_rgb_mid))  # [32, 768]-> 32
		dist_an_ir_mid = torch.diag(euclidean_dist(feat_ir, feat_ir_mid))  # [32, 768]


		dist_an = torch.cat([dist_an_rgb_mid,dist_an_ir_mid], dim=0)
		triplets = dist_ap - dist_an + self.margin

		loss = triplets[triplets > 0].sum() / 64
		return loss, triplets[triplets>0].shape[0]
# class TripletLoss_Balanced(nn.Module):
# 	def __init__(self, margin, feat_norm='yes'):
# 		super(TripletLoss_Balanced, self).__init__()
# 		self.margin = margin
# 		self.feat_norm = feat_norm
# 		if margin >= 0:
# 			self.ranking_loss = nn.MarginRankingLoss(margin=margin)
# 		else:
# 			self.ranking_loss = nn.SoftMarginLoss()
#
# 	def forward(self, global_feat1, global_feat2, target):
#
# 		feat_rgb,feat_ir = global_feat1.chunk(2,0)
#
# 		dist_mat = euclidean_dist(global_feat1, global_feat2)
# 		dist_ap, dist_an = hard_example_mining(dist_mat, target)
#
# 		N = dist_mat.size(0)
# 		# shape [N, N]
#
# 		is_posf = target.expand(N, N).eq(target.expand(N, N).t()).float()
# 		is_neg = target.expand(N, N).ne(target.expand(N, N).t())
#
# 		dist_mat_an = dist_mat[is_neg].view(64, -1)
#
# 		for i in range(64):
# 			idx_an = torch.nonzero(is_posf[i] < 0.5)
# 			feat_an = global_feat1[idx_an].view(56,-1) # 56, 768
#
# 			#idx_ap = torch.nonzero(is_posf[i] > 0.5)  # ap
#
# 			_, idx = torch.topk(dist_mat_an[i].view(1, -1), 2, dim=1, largest=False)
# 			feat_an1 = feat_an[idx.detach().cpu().numpy()]
# 			feat_mean_1 = torch.mean(feat_an1, dim=0).view(1, -1)
#
# 			if i == 0:
# 				feat_mean = feat_mean_1
# 			else:
# 				feat_mean = torch.cat([feat_mean,feat_mean_1],dim=0)
#
# 		feat_rgb_mid,feat_ir_mid = feat_mean.chunk(2,0)
#
# 		dist_an_rgb_mid = torch.diag(euclidean_dist(feat_rgb, feat_rgb_mid)) # [32, 768]-> 32
# 		dist_an_ir_mid = torch.diag(euclidean_dist(feat_ir, feat_ir_mid))  # [32, 768]
#
# 		dist_an = torch.cat([dist_an_rgb_mid,dist_an_ir_mid], dim=0)
# 		triplets = dist_ap - dist_an + self.margin
#
# 		loss = triplets[triplets > 0].sum() / 64
# 		return loss#, triplets[triplets>0].shape[0]
