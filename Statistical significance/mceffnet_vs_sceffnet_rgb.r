library('nonpar')
library('DescTools')


#MC-EffNet-2 with SC-EffNet_RGB
method_a_ss_scEffnet <- matrix(c(694, 79, 116, 57, 532, 144, 83, 138, 557), 3,3)
method_b_ss_scEffnet <- as.table(matrix(c(694, 79, 116, 57, 532, 144, 83, 138, 557), nrow=3))
stuart.maxwell(method_a_ss_scEffnet)
StuartMaxwellTest(method_b_ss_scEffnet)