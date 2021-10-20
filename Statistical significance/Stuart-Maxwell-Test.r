library('nonpar')
library('DescTools')

#MC-EffNet-2 with best baseline 
method_a_ss_baseline <- matrix(c(670, 101, 111, 55, 503, 123, 109, 145, 583), 3,3)
method_b_ss_baseline <- as.table(matrix(c(670, 101, 111, 55, 503, 123, 109, 145, 583), nrow=3))
stuart.maxwell(method_a_ss_baseline)
StuartMaxwellTest(method_b_ss_baseline)

#Psycho Experiments
method_a_ss_psycho <- matrix(c(49, 9, 16, 35, 63, 29, 40, 25, 64), 3,3) 
method_b_ss_psycho <- as.table(matrix(c(49, 9, 16, 35, 63, 29, 40, 25, 64), nrow=3))
stuart.maxwell(method_a_ss_psycho)
StuartMaxwellTest(method_b_ss_psycho)

#MC-EffNet-2 SCEffnet RGB
method_a_ss_scEffnet <- matrix(c(694, 79, 116, 57, 532, 144, 83, 138, 557), 3,3)
method_b_ss_scEffnet <- as.table(matrix(c(694, 79, 116, 57, 532, 144, 83, 138, 557), nrow=3))
stuart.maxwell(method_a_ss_scEffnet)
StuartMaxwellTest(method_b_ss_scEffnet)

