library('nonpar')
library('DescTools')


#Psycho Experiments
method_a_ss_psycho <- matrix(c(49, 9, 16, 35, 63, 29, 40, 25, 64), 3,3) 
method_b_ss_psycho <- as.table(matrix(c(49, 9, 16, 35, 63, 29, 40, 25, 64), nrow=3))
stuart.maxwell(method_a_ss_psycho)
StuartMaxwellTest(method_b_ss_psycho)