library('nonpar')
library('DescTools')

#MC-EffNet-2 with best baseline 
method_a_ss_baseline <- matrix(c(670, 101, 111, 55, 503, 123, 109, 145, 583), 3,3)
method_b_ss_baseline <- as.table(matrix(c(670, 101, 111, 55, 503, 123, 109, 145, 583), nrow=3))
stuart.maxwell(method_a_ss_baseline)
StuartMaxwellTest(method_b_ss_baseline)