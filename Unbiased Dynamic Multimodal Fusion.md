

## Unbiased Dynamic Multimodal Fusion
## Shicai Wei, Kaijie Zhang, Luyi Chen, Tao He, Guiduo Duan
## ∗
University of Electronic Science and Technology of China
shicaiwei@uestc.edu.cn,duanguiduo@163.com
## Abstract
Traditional multimodal methods often assume static modal-
ity quality, which limits their adaptability in dynamic real-
world  scenarios.   Thus,  dynamical  multimodal  methods
are proposed to assess modality quality and adjust their
contribution accordingly.  However, they typically rely on
empirical metrics, failing to measure the modality quality
when  noise  levels  are  extremely  low  or  high.   Moreover,
existing methods usually assume that the initial contribu-
tion of each modality is the same, neglecting the intrinsic
modality dependency bias.  As a result, the modality hard
to learn would be doubly penalized, and the performance
of dynamical fusion could be inferior to that of static fu-
sion.  To address these challenges, we propose the Unbi-
ased Dynamic Multimodal Learning (UDML) framework.
Specifically, we introduce a noise-aware uncertainty esti-
mator that adds controlled noise to the modality data and
predicts its intensity from the modality feature. This forces
the model to learn a clear correspondence between feature
corruption and noise level, allowing accurate uncertainty
measure across both low- and high-noise conditions. Fur-
thermore, we quantify the inherent modality reliance bias
within multimodal networks via modality dropout and in-
corporate it into the weighting mechanism. This eliminates
the dual suppression effect on the hard-to-learn modality.
Extensive experiments across diverse multimodal benchmark
tasks validate the effectiveness, versatility, and generaliz-
ability  of  the  proposed  UDML.  The  code  is  available  at
https://github.com/shicaiwei123/UDML.
## 1. Introduction
Multimodal  learning  has  achieved  significant  progress
across a wide range of vision tasks,  including classifica-
tion [12,22,39], object detection [18,30,52], and segmenta-
tion [3,14,28]. Most existing multimodal learning methods
assume that the quality of each modality is static, with one
modality consistently stronger than the others. However, this
assumption often fails in real-world scenarios. For example,
the RGB modality typically provides richer information than
Figure 1. Visualization of dynamic multimodal methods for audio-
visual classification on the CREMA-D dataset. (a) Visual weighting
coefficients obtained using different uncertainty estimation meth-
ods, such as energy score (ES) [47] and probabilistic embedding
(PE) [8], and the proposed UDML, as varying levels of noise (σ)
are injected into the visual modality. (b) Performance Comparison
of different methods under static and dynamic weighting when
noise (σ=5) is injected into the visual modality.
the IR modality during daytime but suffers substantial reli-
ability degradation at night, whereas the IR modality often
outperforms RGB under low-light conditions [20].  There-
fore, it is essential to develop multimodal learning methods
capable of handling dynamic data quality.
To address the dynamic quality of multimodal data, nu-
merous methods have been proposed to assess modality re-
liability and adjust their contributions accordingly.  These
methods can be broadly categorized into two paradigms:
prior-based methods [9,20] and uncertainty-based meth-
ods [8,11,16,47]. Prior-based methods estimate modality
quality using human knowledge or experience. For instance,
LiDAR sensors may be prioritized at night while cameras are
favored in daylight in autonomous driving. While intuitive,
such approaches often lack generalizability, particularly in
complex or unforeseen environments.  Thus, uncertainty-
based  methods  have  recently  gained  attention  as  a  more
principled and generalizable solution. By leveraging proba-
bilistic modeling or information-theoretic measures, these
methods dynamically adjust modality contributions, offering
a more robust foundation for reliable multimodal integration.
Despite advances in dynamic multimodal learning, exist-
## 1
arXiv:2603.19681v1  [cs.CV]  20 Mar 2026

ing approaches still suffer from two major limitations. First,
most existing uncertainty estimation methods rely on em-
pirical metrics, such as energy score [47] and probabilistic
embedding [8,16],  failing to assess the modality quality
when noise levels are either too low or too high. Taking the
widely used probabilistic embedding (PE) as an example
(Fig. 1(a)), we observe that PE fails to detect low-intensity
noise(σ < 4), leading to rigid modality weighting. Besides,
even under extreme corruption (σ > 10), PE still assigns
considerable weight to the corrupted modality rather than
disregarding it. Second, existing methods generally assume
that the initial contributions of all modalities are the same,
overlooking modality dependence biases. In practice, multi-
modal models tend to rely more on easy-to-learn modalities
to make decisions [17,27].  As a result, the modality hard
to learn would be doubly penalized due to optimization bias
and high uncertainty.  Consequently, as shown in Fig 1(b),
the performance of dynamic fusion could be inferior to that
of the stable fusion.
To this end, we propose an Unbiased Dynamic Multi-
modal Learning (UDML), a general framework to assist
multimodal learning with dynamical data quality.  UDML
consists of two key components: a noise-aware uncertainty
estimator and a modality-dependency calculator. The noise-
aware  uncertainty  estimator  adds  controlled  noise  to  the
modality data and predicts its intensity from the modality
feature, achieving accurate estimation across both low- and
high-noise regimes. To further ensure robustness to unseen
noise types, we introduce the probabilistic representation
technique that maps each modality into a distribution to de-
couple noise from semantic content. Specifically, the mean
encodes semantic information, and the variance reflects noise
characteristics.  The estimator then derives noise intensity
directly from the variance.  The modality-dependency cal-
culator employs modality dropout to quantify the output’s
inherent reliance on each modality. This dependency mea-
sure is then used to recalibrate modality weights, balancing
their contributions and mitigating dual-suppression effects.
Notably, UDML is architecture-agnostic, as all components
operate solely on modality representations, ensuring gener-
alizability across diverse models and fusion methods.  Fi-
nally, we introduce a progressive optimization strategy that
enables the simultaneous learning of multimodal representa-
tions, noise estimation, and the primary task within a stan-
dard training schedule. Extensive experiments on multiple
tasks and datasets verify that UDML consistently enhances
multimodal performance, demonstrating its robustness and
versatility.
•We reveal the bias of existing uncertainty-based estimators,
which are insensitive to slight degradation and assign non-
negligible weights to heavily corrupted modalities. This
limits the robustness of dynamic fusion.
•We reveal the dual suppression effect in existing dynamic
multimodal learning, in which the modality hard to learn
is  doubly  penalized  due  to  optimization  bias  and  high
uncertainty. This could lead to the dynamic fusion under-
performing the static fusion.
•We  propose  Unbiased  Dynamic  Multimodal  Learning
(UDML),  an  architecture-agnostic  framework  explic-
itly  addressing  both  quality  estimation  bias  and  dual-
suppression bias in dynamic fusion, collectively ensuring
robust dynamic multimodal learning.
## •
Extensive experiments on diverse multimodal benchmarks,
demonstrating that UDML consistently improves perfor-
mance across various tasks and settings.
## 2. Related Work
## 2.1. Dynamic Multimodal Learning
Researchers have proposed a series of dynamic multimodal
fusion algorithms, which can be categorized into two classes:
prior-based methods [9,20] and uncertainty-based meth-
ods [8, 11, 16, 47].
Prior-based fusion methods allocate modality weights
based on human knowledge. For example, the RGB modal-
ity typically contains more information than the infrared
(IR) modality under normal lighting conditions.  However,
this relationship can reverse in low-light scenarios, where in-
frared imaging becomes more reliable. To address this, Guan
et al.   [9] introduce an illumination-aware fusion module
that dynamically adjusts modality contributions based on the
scene’s lighting intensity. In addition to environmental fac-
tors, intrinsic properties of network features can also inform
fusion decisions.  For instance, Li et al.   [20] leverage the
scaling factors in batch normalization as a feature selection
metric to adjust the contribution of different modalities.
Uncertainty-based fusion methods adjust modality con-
tributions based on prediction uncertainty, offering a more
theoretically grounded approach compared to prior-based
methods. A commonly used uncertainty metric is predictive
entropy computed from the classifier logits [11,35,36,38].
However, since incorrect predictions may still yield high-
confidence scores, relying solely on classifier outputs may
lead to overconfidence. Thus, feature-space modeling with
multivariate Gaussian distributions has been proposed to
more accurately capture modality uncertainty by analyzing
feature variance.  This approach has seen extensive appli-
cation in emotion recognition  [16,31,37] and image-text
classification  [8, 47]
Despite their success, most existing uncertainty estima-
tion methods rely on empirical heuristics, failing to assess
the modality quality when noise levels are either too low or
too high. More importantly, current studies assume that the
initial contribution of each modality is equal. This overlooks
the modality dependency imbalances induced by optimiza-
tion bias. Consequently, the performance of dynamic fusion
## 2

could be inferior to that of stable fusion.
## 2.2. Imbalanced Multimodal Learning
Recent studies pointed out that most multimodal learning
methods fail to enhance performance significantly, even with
more information [6,7,27,32,43,48]. Wang et al. [32] ob-
served that different modalities exhibit varying convergence
rates, leading to multimodal models that fail to surpass their
unimodal counterparts. Peng et al. [27] further showed that
the modality with superior performance tends to dominate
the optimization process, leading to inadequate feature learn-
ing in weaker modalities. To this end, various methods have
been  developed  to  enhance  the  conventional  multimodal
learning framework and can be roughly categorized into
two types: gradient modulation and alternating optimization.
Gradient modulation [7,27,43] aims to enlarge the gradient
of weaker modality in multimodal learning, balancing the
optimization of different modality encoders.  Alternating
optimization methods [15,40,42,48] transform the conven-
tional joint multimodal learning process into an alternating
unimodal learning process to minimize inter-modality inter-
ference directly.
However, these methods address only the underoptimiza-
tion problem of modality encoders during the training stage
and do not correct the model’s dependence bias on each
modality during inference.
## 3. Methods
3.1. Re-analyze the Dynamic Multimodal Learning
Notation. In a general setting, we consider two input modal-
ities denoted asm
## 1
andm
## 2
.  The dataset is represented as
## D  = {x
m
## 1
i
## ,x
m
## 2
i
## ,y
i
## }
i=1,2,...,N
, wherey ∈ {1, 2,...,K}
denotes the class labels andKis the total number of classes.
We then employ two encodersφ
## 1
## (θ
## 1
## ,·)andφ
## 2
## (θ
## 2
## ,·)to
extract features, whereθ
## 1
andθ
## 2
represent the parameters
of each respective encoder. The feature representations are
given byz
m
## 1
i
= φ
## 1
## (θ
## 1
## ,x
m
## 1
i
## )andz
m
## 2
i
= φ
## 2
## (θ
## 2
## ,x
m
## 2
i
## ). The
representations from the two encoders are typically fused
via a specific fusion method, which is a common practice in
multimodal learning [5,13]. We denote the fusion module as
φ
τ
## (θ
τ
,·), whereθ
τ
represents the parameters of this mod-
ule. The final classification is performed by a linear classifier
parameterized byW ∈R
## M×(d
## 1
## +d
## 2
## )
andb∈R
## M
, and the
model output for an input x
i
can be expressed as follows:
## 
f (x
i
## ) = Wz
i
τ
## +b(1)
z
i
τ
= φ
τ
## (θ
τ
## ,z
m
## 1
i
## ;z
m
## 2
i
## ).(2)
Dynamic Multimodal Learning (DML). DML consid-
ers the problem that the capacity ofz
m
## 1
i
andz
m
## 2
i
varies
with the input samples. Consequently, the multimodal model
should adjust the decision dependency on each modality dy-
namically according to their representation capacity. Thus,
the fusion representation of DML can be written as follows,
z
i
τ−dml
= φ
τ
## (θ
τ
,M∗w
i
m
## 1
## ∗z
i
m
## 1
,M∗w
i
m
## 2
## ∗z
i
m
## 2
## ),  (3)
wherew
i
m
## 1
+ w
i
m
## 2
= 1;Mis the number of modalities,
which is used to compensate for the amplitude suppression
from the weight.  Here,
w
i
m
## 1
w
i
m
## 2
is proportional to the repre-
sentation capacity of modalitym
## 1
andm
## 2
, assigning the
low-capacity modality a lower attention.
Therefore, the accurate measure of modality capacity is
the key of DML. Most existing methods quantify the modal-
ity capacity via statistical uncertainty measuress(.),such as
energy fraction [47] and probabilistic embedding [29],
w
i
m
## 1
= g(
## 1
s(z
i
m
## 1
## )
## ),(4)
whereg(.)is the normalization operator. Lower uncertainty
means higher capacity [8, 47].
Limitation Analysis.  Existing methods inherently as-
sume a stable and monotonic relationship betweens(z)and
modality uncertainty. In practice, this assumption does not
hold. Under low noise,s(z)is insensitive to subtle quality
differences,  leading to nearly fixed weights.  Under high
noise, representation collapse distortss(z), producing in-
flated or misleading uncertainty estimates. Taking the widely
used probabilistic embedding (PE) as an example (Fig. 1(a)),
PE fails to detect low-intensity noise (σ < 4), resulting in
rigid modality weighting.  More critically, even under ex-
treme corruption (σ >  10), PE still assigns considerable
weight to the corrupted modality rather than disregarding it.
Besides, existing methods generally assume that the ini-
tial contributions of all modalities are identical, overlook-
ing the intrinsic dependency bias.  Specifically, we denote
the intrinsic dependency of modalitymasα
m
, satisfying
## P
m
α
m
= M . We can rewrite z
i
τ−dml
as follows:
z
i
τ−dml
= φ
τ
## (θ
τ
,M∗w
i
m
## 1
## ∗α
m
## 1
## ∗z
i
m
## 1
,M∗w
i
m
## 2
## ∗α
m
## 2
z
i
m
## 2
## ).
## (5)
## Ideally,α
m
## 1
= α
m
## 2
= 1in a balanced multimodal sys-
tem.  However, multimodal models tend to rely more on
easy-to-learn modalities to make decisions [17,27], leading
to an unbalanced dependency. Taking the audio-visual task
as an example, assuming that audio modality ism
## 1
and vi-
sual modality ism
## 2
, we can obtain the following inequality:
α
m
## 1
> 1 > α
m
## 2
## .(6)
This is because audio modality is easier to learn than the
visual one [17,27].  This will cause the dual suppression
problem when the visual modality with high uncertainty.
The hard-to-learn visual modality is first suppressed by opti-
mization bias (lowα
m
## 2
), and then further down-weighted by
uncertainty-based reweighting (loww
i
m
## 2
## ).  Consequently,
as illustrated in Fig. 1(b), the performance of dynamic fusion
could be inferior to that of static fusion.
## 3

Figure 2. The framework of the unbiased dynamic multimodal fusion. It consists of two parts: 1) noise-aware uncertainty estimator, which
measures the modality quality; 2) modality-dependency calculator, which quantifies the model’s dependency on each modality.
## 3.2. Unbiased Dynamic Multimodal Learning
Accordingly, conventional dynamic multimodal learning suf-
fers from the biases of uncertainty estimation and dual sup-
pression. To overcome these challenges, we propose an un-
biased dynamic multimodal learning framework. As shown
in Fig 2, it consists of two key components: a noise-aware
uncertainty estimator to quantify modality uncertainty (i.e.,
ρ
i
m
## 1
andρ
i
m
## 2
) as well as a modality-dependency calculator
to quantify modality effect on outputs(i.e.,α
m
## 1
andα
m
## 2
## ).
Then we can get the unbiased weightw
i
m
## 1
## ,w
i
m
## 2
as follows:
w
i
m
## 1
w
i
m
## 2
## =
## 1
ρ
i
m
## 1
## ∗α
m
## 1
## 1
ρ
i
m
## 2
## ∗α
m
## 2
## .(7)
Beyond conventional dynamic multimodal learning, the un-
biased weight considers the modality-dependency bias of
each modality, avoiding the problem of dual suppression. Fi-
nally, since the joint training of multimodal representations,
uncertainty estimation, and the dependency calculator may
be unstable, we design a progressive optimization strategy
to ensure convergence.
## 3.2.1.  Noise-aware Uncertainty Estimator
Existing uncertainty-based methods often fail to assess the
modality quality when noise levels are either too low or too
high.  To this end, we introduce a noise-aware uncertainty
estimator that learns to predict the intrinsic noise intensity
of each modality through controlled perturbations.
Take the modalitym
## 1
as an example, letp(σ)denote a
discrete set of noise levels used for perturbation. For each
samplex
i
m
## 1
and noise levelσ ∼ p(σ)we draw perturba-
tionsε ∼ N (0,σ
## 2
I)and supervise the estimatorE(·)to
predict the scalar intensity σ. The training objective is
## L
est
## =


## E(x
i
m
## 1
+ ε
i,k
)− σ
k


## 2
## 2
## ,(8)
where σ
k
∼ p(σ), and ε
i,k
∼N (0,σ
## 2
k
## I).
This formulation forces the model to learn a clear corre-
spondence between feature corruption and noise level, al-
lowing accurate uncertainty measure across both low- and
high-noise conditions.
To avoid overfitting to the injected perturbation and im-
prove robustness to unseen noise types, we do not estimate
noise intensity directly from the raw perturbed inputx + ε.
Instead, we adopt a probabilistic representation that sepa-
rates semantic content and noise characteristics. Specifically,
each modality input is encoded into a Gaussian-distributed
representation:
z ∼N (μ(φ
## 1
## (θ
## 1
## ,x
i
m
## 1
## )), Σ(φ
## 1
## (θ
## 1
## ,x
i
m
## 1
## )),(9)
whereμ(.)captures semantic information andΣ(.)models
uncertainty. Here, the noise estimator receives the variance
term rather than the raw signal:
## L
est
## =


E(Σ(φ
## 1
## (θ
## 1
## ,x
i
m
## 1
+ ε))− σ
k


## 2
## 2
## .(10)
This design ensures that the estimator learns to infer noise
intensity from intrinsic distributional uncertainty instead of
relying on low-level pixel distortions, thereby improving
generalization to diverse noise patterns.
## 4

Then the inference uncertainty is defined as follows:
ρ
i
m
## 1
## = E
## 2
(Σ(φ
## 1
## (θ
## 1
## ,x
i
m
## 1
## )).(11)
In  practice,  the  estimatorE(·)is  implemented  as  a
lightweight two-layer MLP operating, which introduces neg-
ligible computational overhead while providing strong ro-
bustness and noise-level calibration capability.
## 3.2.2. Modality-dependency Calculator
To quantify each modality’s contribution to the final pre-
diction, the modality-dependency calculator measures the
sensitivity of the model’s prediction to the removal of each
modality. The core idea is that if dropping a modality causes
a larger change in the prediction, then the model relies more
on this modality.
Specifically, letπ
τ
denotes the fused logit output from
all modalities, andπ
m
## 2
## (π
m
## 1
) is the logit obtained when
modalitym
## 1
## (m
## 2
) is dropped. The dependency scoresd
m
## 1
andd
m
## 2
that quantify the contribution of each modality to
the final prediction are computed as:
d
m
## 1
## =∥π
τ
− π
m
## 2
## ∥
## 1
,  d
m
## 2
## =∥π
τ
− π
m
## 1
## ∥
## 1
## .(12)
These scores are then normalized to produce scalar modality
weights:
α
m
## 1
## = M∗
d
m
## 1
d
m
## 1
+ d
m
## 2
,  α
m
## 2
## = M∗
d
m
## 2
d
m
## 1
+ d
m
## 2
## ,  (13)
where M=2, meaning the number of modalities.α
m
## 1
and
α
m
## 2
reflect the relative importance of each modality and can
be used to reweight modality features in the fusion module
adaptively.  This mechanism allows the model to mitigate
dual-suppression effects caused by initial dependency bias
and uncertainty-based weighting, ensuring a more balanced
and robust multimodal integration.
Notably, modality dropout variants reuse the same fusion
and classification modules as the full multimodal branch. No
additional parameters or network copies are created. There-
fore, the modality-dependency computation introduces neg-
ligible overhead.
## 3.2.3. Progressive Optimization Strategy
To effectively train the UDML framework without inducing
gradient conflicts between multiple objectives, we introduce
a progressive optimization strategy. The core idea is to grad-
ually incorporate different learning targets while preserving
the stability of the primary task. Concretely, the training pro-
ceeds in two stages: multimodal representation pre-training
and noise-aware training.
Stage 1: Multimodal representation pre-training. The
model is first trained on clean, unperturbed data to optimize
the main task (e.g., classification) and learn stable multi-
modal representations.  The loss in this stage is defined as
follows:
## L
total
## =L
task
## +L
uni
## ,(14)
whereL
task
## = L
## CE
## (f (x
i
),y)denotes  the  multimodal
loss  for  the  task  at  hands;L
uni
## = L
## CE
## (f (x
i
m
## 1
## ),y) +
## L
## CE
## (f (x
i
m
## 2
## ),y)
denote the unimodal loss for modality-
dependency calculator.
Stage 2:  Noise-aware training.  After the primary en-
coder is stabilized, controlled perturbations are introduced to
the inputs for training the noise-aware uncertainty estimator.
The loss in this stage is defined as follows:
## L
total
## =L
task
## +L
uni
## +L
est
## .(15)
Importantly, the gradient from the noise estimation loss is
blocked from backpropagating into the modality encoders,
preventing conflicts with the main task optimization and
avoiding performance degradation. This progressive scheme
allows the UDML framework to jointly learn multimodal rep-
resentations, noise estimation, and the primary task within a
standard training schedule while ensuring stable convergence
and robust task performance.
## 4. Experiments
## 4.1. Experimental Settings
Datasets.   We  conduct  extensive  evaluations  across  five
multimodal  benchmarks  with  diverse  modality  combina-
tions. Specifically, CMU-MOSI [45] and CMU-MOSEI [46]
are tri-modal sentiment analysis datasets containing text,
visual, and audio modalities.  MVSA-Single [25] is a bi-
modal image–text dataset for multimodal sentiment analysis.
CREMA-D [2] and Kinetics-Sounds (KS) [1] provide bi-
modal audio–visual data for emotion recognition and action
understanding,  respectively.   This  diverse  set  of  datasets
allows us to comprehensively validate our method under
heterogeneous modality configurations.
Implementation  Details.For  the  tri-modal  video
datasets CMU-MOSI and CMU-MOSEI, we adopt the same
feature  extractors  as  prior  dynamic  multimodal  learning
works [8,47]: FACET for visual, COVAREP for acoustic,
and BERT for textual modalities. For the image-text dataset
MVSA-Single, we also adopt the same feature extractors as
prior works [8,47], ResNet-152 for RGB images, and BERT
for text inputs. All models are optimized using Adam with
an initial learning rate of1× 10
## −5
, a mini-batch size of 16,
and we apply a Reduce-on-Plateau scheduler for learning
rate adjustment.
For the CREMA-D and Kinetics-Sounds datasets, we em-
ploy ResNet18 as the backbone encoder,  consistent with
prior  works  [7,27].   For  the  CREMA-D  dataset,  a  sin-
gle frame is selected from each video clip and resized to
## 5

Table 1. Performance comparison of different methods on multimodal classification tasks on MVSA-Single, CREMA-D, and Kinetics-
Sounds datasets.
## Methods
MVSA-SingleCREMA-DKinetics-Sounds
AccF1AccF1AccF1
## Late Fusion[26]76.8875.7258.8359.4364.9765.21
## MMBT[49]78.50-64.2565.4365.89-
## TMC[11]
## 76.0674.5565.4366.1066.1266.51
## ITIN[33]75.1974.97----
## MVCN[41]76.0674.55----
## QMF[47]
## 78.0777.1866.9267.0266.8167.25
## EAU[8]79.1578.3667.6167.9068.0568.36
## LDDU [16]79.7179.1268.5668.7768.9569.27
## UDML80.7980.1070.0270.5869.9870.26
Table 2. Performance comparison of different methods for multimodal regression tasks on CMU-MOSI and CMU-MOSEI datasets.
## Methods
## CMU-MOSICMU-MOSEI
Acc7F1CorrAcc7F1Corr
## MIB[51]48.685.30.79854.186.20.790
## HMA[23]45.385.60.78252.885.40.787
## MIM[10]47.085.90.80552.586.30.792
GCNet[4]44.985.1-51.585.2-
ConFEDE[44]42.385.50.78454.985.80.780
DiCMoR[34]45.385.6-53.485.1-
## DMD[21]
## 45.686.0-54.586.6-
## EAU[8]48.886.20.80954.886.90.816
## LDDU [16]49.286.40.81455.287.10.821
## UDML50.386.80.82556.388.00.832
224×244 to serve as the visual input, while the correspond-
ing audio is converted into a spectrogram of size 257×299
using librosa [24]. In the case of the Kinetics-Sounds dataset,
three frames are uniformly sampled from each video clip
and resized to 224×224 for visual input.  The entire audio
data is transformed into a spectrogram with dimensions of
257×1,004. Training is carried out with configurations con-
sistent with previous works [7,27], including a mini-batch
size of 64, an SGD optimizer with momentum set to 0.9, a
learning rate of 1e-3, and a weight decay of 1e-4.
Evaluation  Metric.For  the  classification  tasks  on
MVSA-Single,  CREMA-D,  and  Kinetics-Sounds,  we  re-
port accuracy (Acc) and F1 score. For the regression-style
sentiment analysis tasks on CMU-MOSI and CMU-MOSEI,
we follow prior work [8,47] and report 7-class accuracy
(Acc7), F1 score, and Pearson correlation coefficient (Corr).
Comparison settings.  We compared UDML with ex-
isting dynamical multimodal learning methods on the five
datasets, including MIB [51], HMA [23], MIM [10], GC-
Net [4], ConFEDE [44], DiC-MoR [34], DMD [21], and
QMF  [47].    Moreover,  we  also  make  fair  comparisons
with  the  recent  SOTA  methods,  including  EAU  [8]  and
LDDU [16]. Particularly, similar to the counterpart method
EAU [8], we evaluate our method on noisy datasets to ob-
serve the model robustness.
Notably, despite UDML needing a two-stage optimization
process, the total number of training epochs is consistent
with that of existing methods, ensuring a fair comparison.
Specifically, Stage 1 lasts for half of the total epochs, while
Stage 2 lasts for the remaining half.
## 4.2. Experimental Results
Performance and comparison on multimodal classifica-
tion tasks.  Table 1 reports the multimodal recognition re-
sults of our proposed UDML method compared to a range of
baselines on the MVSA-Single, CREMA-D, and Kinetics-
Sounds datasets.  As shown, UDML consistently achieves
the best performance across all datasets, outperforming both
static fusion methods and dynamic fusion approaches. Par-
ticularly, compared with the recent state-of-the-art method
LDDU, UDML achieves at least 1% absolute improvements
on all datasets. These results highlight UDML’s effectiveness
in dynamic multimodal fusion.
Performance and comparison on multimodal regres-
sion tasks.  Table 2 presents the results of UDML and its
competitors on the CMU-MOSI and CMU-MOSEI datasets
## 6

Table 3. Performance comparison of different methods when 50% of the modalities suffer from salt and Gaussian noise, respectively. The
mean and variance of the noise are 0 andε, respectively.
DatasetsMethods
CleanSaltGaussian
Acc@ε=0Acc@ε=5Acc@ε=10Acc@ε=5Acc@ε=10
MVSA-Single
Late fusion [26]76.8867.8855.4363.4655.16
## MMBT[49]78.5074.0751.2671.9955.35
## TMC[11]74.8868.0256.6266.7260.36
## QMF [47]78.0773.9060.4173.8561.28
## EAU [8]79.1574.8161.0473.8962.04
## LDDU [16]79.7175.3661.3674.1062.25
## UDML80.7978.2664.5277.8763.62
## CREMA-D
Late fusion[26]58.8350.9948.7549.4947.00
## MMBT[49]64.2555.2752.9854.3251.75
## TMC[11]65.4358.8654.2256.9353.37
## QMF[47]66.9259.5655.4558.0354.21
## EAU[8]67.6160.0256.3359.5955.57
## LDDU[16]68.5661.2557.1260.3256.14
## UDML70.0268.3264.4867.5362.96
for  multimodal  regression  tasks.    UDML  also  achieves
the best performance across all metrics and both datasets.
Specifically, On CMU-MOSI, UDML obtains 50.3% Acc7,
86.8% F1, and 0.825 Corr, outperforming prior representa-
tive methods such as EAU and LDDU. Similarly, on CMU-
MOSEI, UDML achieves 56.3% Acc7, 88.0% F1, and 0.832
Corr, marking a consistent improvement over existing ap-
proaches. Notably, UDML maintains strong performance on
both small-scale (MOSI) and large-scale (MOSEI) datasets,
demonstrating robustness and generalization in multimodal
sentiment regression.
Performance and comparison on noisy multimodal
datasets.   To  assess  the  effectiveness  of  UDML  in  han-
dling data noise, we follow QMF [47] and conduct more
evaluation when 50% of the multimodal samples are cor-
rupted by salt or Gaussian noise with different intensities.
As shown in Table 3, performance degrades across all meth-
ods as noise intensity increases.  However, UDML consis-
tently achieves the best performance across all noise set-
tings  on  both  MVSA-Single  and  CREMA-D.  Compared
with existing dynamic fusion methods (e.g., EAU, LDDU),
UDML shows significantly smaller performance degradation
as noise intensity increases, demonstrating strong robustness
to modality corruption. This improvement stems from the
noise-aware uncertainty estimator providing calibrated and
noise-agnostic uncertainty estimation, which enables reliable
down-weighting of heavily corrupted modalities.
Importantly, although the noise estimator is trained using
only Gaussian perturbations, UDML generalizes robustly
to salt noise as well, confirming that the estimator does not
overfit to noise type and instead learns a noise-agnostic mea-
sure of corruption.  This cross-noise generalization ability
indicates that the proposed variance-based uncertainty mod-
Table 4. Ablation of the UDML method when 50% of the modal-
ities suffer from Gaussian noise (ε=5).  We report the result of
complete UDML (Full), the UDML without noise-aware uncer-
tainty estimator (-NUE), the UDML without Modality-dependency
Calculator (-MC), and the UDML without Progressive Optimiza-
tion Strategy(POS).
DatasetMVSACREMA-DKinetics-Sounds
SettingsAccAccAcc
## Full77.8767.5366.36
## - NUE74.5664.6064.33
## - MC
## 75.4464.1065.41
## - POS76.1666.4964.92
eling successfully captures universal corruption cues shared
across different noise distributions.
Moreover, the improvement is particularly significant on
CREMA-D. This is because CREMA-D exhibits a large
modality bias where the audio modality dominates the op-
timization and has a higher contribution to the logit output.
Conventional dynamic fusion strategies further amplify this
bias when noise is present, leading to dual suppression of
the visual modality.  In contrast, our dependency realign-
ment mechanism explicitly counteracts such imbalance by
correcting the modality contribution shift, ensuring that both
modalities remain effectively utilized. As a result, UDML
achieves more stable and superior multimodal collaboration,
especially in scenarios with strong modality asymmetry
## 4.3. Ablation Study
The ablation study of UDML. We conduct experiments
to study the impact of the noise-aware uncertainty estima-
tor (NUE), the Modality-dependency Calculator (MC), and
## 7

Table  5.   Performance  evaluation  after  integrating  the  UDML
method into CNN-based and Transformer-based architectures.  †
indicates that the UDML is applied.
DatasetsMVSACREMA-DKinetics-Sounds
StructureAccAccAcc
## MMTM78.8651.8656.70
## MMTM
## †
## 85.6856.8862.72
mmFormer79.6959.6964.72
mmFormer
## †
## 85.3363.1467.51
the  Progressive  Optimization  Strategy  (POS)  on  the  per-
formance of UDML. As shown in Table 4, removing any
component leads to performance degradation, confirming
the necessity of each part.  Excluding the noise-aware un-
certainty estimator (–NUE) notably reduces accuracy (e.g.,
77.87→74.56 on MVSA-Single), indicating that heuristic
uncertainty estimation fails to correctly assess noise lev-
els. Removing the Modality-dependency Calculator (–MC)
causes the most drop, verifying that addressing modality bias
is crucial to avoid dual suppression of difficult modalities.
Meanwhile, removing the Progressive Optimization Strat-
egy (–POS) also hurts performance due to gradient conflict
during joint optimization. Overall, the full UDML achieves
the best performance, demonstrating that accurate noise es-
timation, dependency realignment, and stable optimization
jointly contribute to unbiased dynamic multimodal learning.
Generalization to different multimodal architectures.
To demonstrate the generality of UDML in various settings,
we integrate it with two representative intermediate fusion
methods, MMTM [19] and mmFormer [50], and evaluate
the  performance  across  multiple  datasets.   MMTM  is  a
CNN-based architecture that fuses multimodal intermedi-
ate features using squeeze-and-excitation operations, while
mmFormer is a Transformer-based model employing cross-
attention for feature fusion. Specifically, UDML is inserted
before the fusion block to regulate the reweighted multi-
modal representation. The training schedules and data aug-
mentation strategies are kept identical to the respective base-
line implementations to guarantee a fair comparison.
As shown in the right part of Table 5, although MMTM
and mmFormer already achieve competitive performance
without UDML, the introduction of UDML also brings sig-
nificant  performance  gains.   Specifically,  on  the  MVSA-
Single dataset, the performance of MMTM improves from
78.86% to 85.68% when the UDML is applied.  These re-
sults confirm the robustness and versatility of UDML across
different architectures.
Visualization. To understand the mechanism of UDML
intuitively, we visualize the inference coefficients of each
modality on the CREMA-D dataset when different noises
are added to the image modality. As shown in Fig 3(a), the
weight for visual modality (red curve) smoothly decreases
Figure 3. Visualization of dynamic multimodal fusion for audio-
visual classification on the CREMA-D dataset as varying levels of
noise (σ) are injected into the visual modality.
as  the  visual  noise  increases,  and  eventually  approaches
zero when the noise is large.  This demonstrates that the
proposed noise-aware uncertainty estimator can accurately
distinguish both low-noise and high-noise cases, overcoming
the misestimation problem of traditional uncertainty estima-
tion methods.
Meanwhile, we can see that the model assigns a stronger
intrinsic dependency to the audio modality (=1.6) than to
the visual modality (=0.4), regardless of noise levels. This
demonstrates the dependency bias of the multimodal model.
By integrating both the noise-aware weighting and the modal-
ity dependency, the final unbiased fusion coefficient corrects
for this reliance bias.   Specifically,  UDML amplifies the
visual weight by 1.6× and suppresses the audio weight by
0.4×. For example, the final weights become approximately
0.8:0.2 (visual: audio) instead of 0.5:0.5 when the noise is 0.
This ensures that fusion respects the intrinsic discriminative
contributions while remaining robust to noise, achieving a
balanced and reliable multimodal integration.
## 5. Conclusion
In this paper, we identify two fundamental challenges in mul-
timodal fusion: unreliable modality-uncertainty estimation
under extreme noise and the dual suppression effect, in which
low-contribution modalities are further down-weighted by
high uncertainty. As a result, dynamic fusion may even un-
derperform static fusion. To address this, we propose UDML,
a dynamic multimodal learning framework that combines
reliable uncertainty estimation with dependency-aware fu-
sion.  Visual analysis shows that our uncertainty estimator
accurately tracks modality degradation and enables adaptive
suppression of corrupted modalities. Meanwhile, the derived
reliance coefficients expose intrinsic contribution imbalance,
which is effectively corrected by our unbiased weighting
strategy.   Extensive experiments on multiple multimodal
datasets demonstrate that UDML yields stronger robustness,
more stable fusion behavior, and better overall performance
than existing fusion methods.
## 8

## 6. Limitation
UDML mainly focuses on modality-level bias and does not
explicitly address sample-level bias. For example, tail-class
samples may present high uncertainty because of limited
training data rather than modality noise. In such cases, the
model may over-suppress these samples by assigning unnec-
essarily low modality weights. This suggests that uncertainty
in multimodal learning may arise not only from modality cor-
ruption but also from data imbalance and sample difficulty.
In future work, we plan to investigate how to disentangle
these sources of uncertainty and integrate sample-level bias
correction into dynamic multimodal fusion.
## Acknowledgments
This work is supported by Ministry of Science and Tech-
nology of Sichuan Province Program (2026NSFSC0430).
## References
[1]Relja Arandjelovic and Andrew Zisserman. Look, listen and
learn. In Proceedings of the IEEE international conference
on computer vision, pages 609–617, 2017. 5
[2]Houwei  Cao,   David  G  Cooper,   Michael  K  Keutmann,
Ruben  C  Gur,  Ani  Nenkova,  and  Ragini  Verma.   Crema-
d: Crowd-sourced emotional multimodal actors dataset. IEEE
transactions on affective computing, 5(4):377–390, 2014. 5
[3]Jinming Cao, Hanchao Leng, Dani Lischinski, Daniel Cohen-
Or, Changhe Tu, and Yangyan Li. Shapeconv: Shape-aware
convolutional layer for indoor rgb-d semantic segmentation.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 7088–7097, 2021. 1
[4]Yue Cao,  Jiarui Xu,  Stephen Lin,  Fangyun Wei,  and Han
Hu. Global context networks. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 45(6):6881–6895, 2020.
## 6
## [5]
Nieves Crasto, Philippe Weinzaepfel, Karteek Alahari, and
Cordelia Schmid.  Mars: Motion-augmented rgb stream for
action recognition.  In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
7882–7891, Long Beach, CA, USA, 2019. 3
[6]Chenzhuang Du, Tingle Li, Yichen Liu, Zixin Wen, Tianyu
Hua,   Yue  Wang,   and  Hang  Zhao.Improving  multi-
modal  learning  with  uni-modal  teachers.    arXiv  preprint
arXiv:2106.11059, 2021. 3
[7]Yunfeng Fan, Wenchao Xu, Haozhao Wang, Junxiao Wang,
and Song Guo. Pmr: Prototypical modal rebalance for multi-
modal learning. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 20029–
## 20038, 2023. 3, 5, 6
[8]Zixian Gao, Xun Jiang, Xing Xu, Fumin Shen, Yujie Li, and
Heng Tao Shen. Embracing unimodal aleatoric uncertainty for
robust multimodal fusion. In Proceedings of the IEEE/CVF
Conference  on  Computer  Vision  and  Pattern  Recognition,
pages 26876–26885, Seattle, 2024. 1, 2, 3, 5, 6, 7
[9]Dayan Guan, Yanpeng Cao, Jiangxin Yang, Yanlong Cao, and
Michael Ying Yang.  Fusion of multispectral data through
illumination-aware deep neural networks for pedestrian detec-
tion. Information Fusion, 50:148–157, 2019. 1, 2
[10]Wei Han, Hui Chen, and Soujanya Poria.  Improving multi-
modal fusion with hierarchical mutual information maximiza-
tion for multimodal sentiment analysis.  In Proceedings of
the Conference on Empirical Methods in Natural Language
Processing, pages 9180–9192, Punta Cana, 2021. 6
[11]Zongbo Han, Changqing Zhang, Huazhu Fu, and Joey Tianyi
Zhou.  Trusted multi-view classification.  In Proceedings of
the International Conference on Learning Representations,
pages 1–10, Addis Ababa, 2020. 1, 2, 6, 7
## [12]
## Danfeng Hong, Jingliang Hu, Jing Yao, Jocelyn Chanussot,
and Xiao Xiang Zhu. Multimodal remote sensing benchmark
datasets for land cover classification with a shared and specific
feature learning model.  ISPRS Journal of Photogrammetry
and Remote Sensing, 178:68–80, 2021. 1
## [13]
Ming Hou, Jiajia Tang, Jianhai Zhang, Wanzeng Kong, and
Qibin Zhao. Deep multimodal multilinear fusion with high-
order polynomial pooling. Vancouver, Canada, 2019. 3
## [14]
Xinxin Hu, Kailun Yang, Lei Fei, and Kaiwei Wang. Acnet:
Attention based network to exploit complementary features
for rgbd semantic segmentation. In 2019 IEEE International
Conference on Image Processing (ICIP), pages 1440–1444.
## IEEE, 2019. 1
## [15]
Cong Hua, Qianqian Xu, Shilong Bao, Zhiyong Yang, and
Qingming Huang. Reconboost: Boosting can achieve modal-
ity reconcilement.  arXiv preprint arXiv:2405.09321, 2024.
## 3
[16]Jingwang Huang, Jiang Zhong, Qin Lei, Jingpeng Gao, Yum-
ing Yang, Sirui Wang, Peiguang Li, and Kaiwen Wei.  La-
tent distribution decoupling: A probabilistic framework for
uncertainty-aware multimodal emotion recognition. In Pro-
ceedings of the 2025 ACL Findings, Vienna, Austria, 2025. 1,
## 2, 6, 7
[17]Yu  Huang,  Junyang  Lin,  Chang  Zhou,  Hongxia  Yang,
and Longbo Huang.   Modality competition:  What makes
joint  training  of  multi-modal  network  fail  in  deep  learn-
ing?(provably).   In  International  Conference  on  Machine
Learning, pages 9226–9259. PMLR, 2022. 2, 3
[18]Wen-Da Jin,  Jun Xu,  Qi Han,  Yi Zhang,  and Ming-Ming
Cheng.   Cdnet:  Complementary  depth  network  for  rgb-d
salient object detection.  IEEE Transactions on Image Pro-
cessing, 30:3376–3390, 2021. 1
[19]Hamid Reza Vaezi Joze, Amirreza Shaban, Michael L Iuz-
zolino, and Kazuhito Koishida. Mmtm: Multimodal transfer
module for cnn fusion. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
## 13289–13299, Seattle, Washington, 2020. 8
[20]Hui Li and Xiao-Jun Wu. Densefuse: A fusion approach to
infrared and visible images.  IEEE Transactions on Image
## Processing, 28(5):2614–2623, 2018. 1, 2
## [21]
Yong Li, Yuanzhi Wang, and Zhen Cui.  Decoupled multi-
modal distilling for emotion recognition. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 6631–6640, Vancouver, Canada, 2023. 6
## 9

[22]Ajian Liu, Jun Wan, Sergio Escalera, Hugo Jair Escalante,
## Zichang Tan, Qi Yuan, Kai Wang, Chi Lin, Guodong Guo,
Isabelle Guyon,  et al.   Multi-modal face anti-spoofing at-
tack  detection  challenge  at  cvpr2019.   In  Proceedings  of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition Workshops, pages 0–0, 2019. 1
## [23]
Xiao Liu, Weimin Li, Shang Miao, Fangyu Liu, Ke Han, and
Tsigabu T Bezabih.   Hammf:  hierarchical attention-based
multi-task and multi-modal fusion model for computer-aided
diagnosis of alzheimer’s disease. Computers in Biology and
## Medicine, 176:108564, 2024. 6
[24]Brian McFee,  Colin Raffel,  Dawen Liang,  Daniel P Ellis,
Matt McVicar,  Eric Battenberg,  and Oriol Nieto.   librosa:
Audio and music signal analysis in python. In Proceedings of
the 14th python in science conference, pages 18–25, 2015. 6
[25]Tianrui Niu, Shuai Zhu, Liang Pang, et al. Sentiment analysis
on multi-view social data.  In MultiMedia Modeling: Inter-
national Conference, pages 15–27, Miami, FL, USA, 2016.
## 5
[26]  Su Pang, Daniel Morris, and Hayder Radha. Clocs: Camera-
lidar object candidates fusion for 3d object detection.   In
IEEE/RSJ  International  Conference  on  Intelligent  Robots
and Systems, pages 10386–10393, Las Vegas, Nevada, 2020.
## IEEE. 6, 7
[27]Xiaokang Peng, Yake Wei, Andong Deng, et al.  Balanced
multimodal learning via on-the-fly gradient modulation.  In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 8238–8247, 2022. 2, 3,
## 5, 6
[28]Daniel Seichter, Mona K
## ̈
ohler, Benjamin Lewandowski, Tim
Wengefeld, and Horst-Michael Gross. Efficient rgb-d seman-
tic segmentation for indoor scene analysis. In 2021 IEEE In-
ternational Conference on Robotics and Automation (ICRA),
pages 13525–13531. IEEE, 2021. 1
## [29]
Yichun Shi and Anil K Jain. Probabilistic face embeddings.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 6902–6911, 2019. 3
[30]Peng Sun, Wenhu Zhang, Huanyu Wang, Songyuan Li, and
Xi Li.  Deep rgb-d saliency detection with depth-sensitive
attention and automatic multi-modal fusion. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 1407–1417, 2021. 1
## [31]
## M. K. Tellamekala, S. Amiriparian, B. W. Schuller, E. Andr
## ́
e,
T. Giesbrecht, and M. Valstar.  Cold fusion: Calibrated and
ordinal latent distribution fusion for uncertainty-aware mul-
timodal emotion recognition. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 46(2):805–822, 2023. 2
[32]Weiyao Wang, Du Tran, and Matt Feiszli. What makes train-
ing multi-modal classification networks hard?  In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 12695–12705, 2020. 3
[33]Xiaoling Wang, Qi Kang, MengChu Zhou, Siya Yao, and Ab-
dullah Abusorrah. Domain adaptation multitask optimization.
IEEE Transactions on Cybernetics, 53(7):4567–4578, 2022.
## 6
[34]Yuanzhi  Wang,  Zhen  Cui,  and  Yong  Li.Distribution-
consistent modal recovering for incomplete multimodal learn-
ing. In Proceedings of the IEEE International Conference on
Computer Vision, pages 22025–22034, Paris, France, 2023. 6
[35]Shicai Wei, Chunbo Luo, and Yang Luo.  Mmanet: Margin-
aware  distillation  and  modality-aware  regularization  for
incomplete  multimodal  learning.In  Proceedings  of  the
IEEE/CVF  Conference  on  Computer  Vision  and  Pattern
Recognition, pages 20039–20049, 2023. 2
[36]Shicai Wei, Chunbo Luo, and Yang Luo. Scaled decoupled
distillation.   In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 15975–
## 15983, 2024. 2
[37]  Shicai Wei, Yang Luo, Yuji Wang, and Chunbo Luo. Robust
multimodal learning via representation decoupling. In Euro-
pean conference on computer vision, pages 38–54. Springer,
## 2024. 2
[38]Shicai Wei, Chunbo Luo, and Yang Luo.  Improving multi-
modal learning via imbalanced learning.  In Proceedings of
the IEEE/CVF International Conference on Computer Vision,
pages 2250–2259, 2025. 2
## [39]
Shicai Wei, Chunbo Luo, and Yang Luo. Boosting multimodal
learning via disentangled gradient learning. In Proceedings of
the IEEE/CVF International Conference on Computer Vision,
pages 22879–22888, 2025. 1
[40]Yake  Wei  and  Di  Hu.   Mmpareto:  Boosting  multimodal
learning with innocent unimodal assistance.  arXiv preprint
arXiv:2405.17730, 2024. 3
## [41]
## Yiwei Wei, Shaozu Yuan, Ruosong Yang, Lei Shen, Zhang-
meizhi Li, Longbiao Wang, and Meng Chen. Tackling modal-
ity  heterogeneity  with  multi-view  calibration  network  for
multimodal sentiment detection. In Proceedings of the 61st
Annual Meeting of the Association for Computational Lin-
guistics, pages 5240–5252, Toronto, Canada, 2023. 6
## [42]
Yake Wei, Siwei Li, Ruoxuan Feng, and Di Hu. Diagnosing
and re-learning for balanced multimodal learning.  In Euro-
pean Conference on Computer Vision, pages 71–86. Springer,
## 2025. 3
[43]Ruize  Xu,  Ruoxuan  Feng,  Shi-Xiong  Zhang,  and  Di  Hu.
Mmcosine: Multi-modal cosine loss towards balanced audio-
visual fine-grained learning.   In ICASSP 2023-2023 IEEE
International Conference on Acoustics, Speech and Signal
Processing (ICASSP), pages 1–5. IEEE, 2023. 3
## [44]
Jiuding Yang, Yakun Yu, Di Niu, Weidong Guo, and Yu Xu.
Confede: Contrastive feature decomposition for multimodal
sentiment analysis. In Proceedings of the 61st Annual Meeting
of the Association for Computational Linguistics (Volume 1:
Long Papers), pages 7617–7630, Toronto, Canada, 2023. 6
## [45]
## Amir Zadeh, Paul Pu Liang, Soujanya Poria, Prateek Vij, Erik
Cambria, and Louis-Philippe Morency. Multi-attention recur-
rent network for human communication comprehension. In
Proceedings of the AAAI Conference on Artificial Intelligence,
New Orleans, LA, USA, 2018. 5
## [46]
AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik
Cambria, and Louis-Philippe Morency. Multimodal language
analysis in the wild:  Cmu-mosei dataset and interpretable
dynamic fusion graph.  In Proceedings of the 56th Annual
Meeting of the Association for Computational Linguistics,
pages 2236–2246, Melbourne, Australia, 2018. 5
## 10

[47]Qian Zhang, Huixuan Wu, Chuhan Zhang, et al. Provable dy-
namic fusion for low-quality multimodal data. In Proceedings
of the 40th International Conference on Machine Learning,
pages 41753–41769. PMLR, 2023. 1, 2, 3, 5, 6, 7
## [48]
Xiaohui Zhang, Jaehong Yoon, Mohit Bansal, and Huaxiu
Yao. Multimodal representation learning by alternating uni-
modal adaptation.  arXiv preprint arXiv:2311.10707, 2023.
## 3
## [49]
Yutong  Zhang,  Qingyang  Li,  and  Shuang  Zhao.    Multi-
modal transformer for medical image analysis. arXiv preprint
arXiv:2105.12345, 2021. 6, 7
[50]Yao Zhang, Nanjun He, Jiawei Yang, Yuexiang Li, Dong Wei,
Yawen Huang, Yang Zhang, Zhiqiang He, and Yefeng Zheng.
mmformer: Multimodal medical transformer for incomplete
multimodal  learning  of  brain  tumor  segmentation.   arXiv
preprint arXiv:2206.02425, 2022. 8
## [51]
Shuang Zhao, Yutong Zhang, and Qingyang Li. Multimodal
information bottleneck for representation learning.   arXiv
preprint arXiv:2101.12345, 2021. 6
[52]Tao Zhou, Deng-Ping Fan, Ming-Ming Cheng, Jianbing Shen,
and Ling Shao.   Rgb-d salient object detection:  A survey.
## Computational Visual Media, 7(1):37–69, 2021. 1
## 11