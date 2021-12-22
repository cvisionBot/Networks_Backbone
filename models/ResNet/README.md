# Deep Residual Learning for Image Recognition

> **Abstract**
> 
- Network 구조를 이전과 달리 깊게 쌓아 학습을 쉽게 하려고 `Residual Learning Framework` 발표한다.
- 다른 기능들은 사용하지 않으며, Layer Input에 `Learning Residual Functions` 적용하면서 Layer의 구조를 재구성하는  방법이다.
- Depth (of Representations)는 많은 Visual Recognition Tasks에서 중요하다.

<aside>
📌 Network의 Depth 늘릴 경우, Gradient Vanishing 문제가 발생한다. 따라서 깊게 쌓기 위한 방향성의 연구 토대가 되었다.

</aside>

> **Introduction**
> 
- Deep Convolutional Neural Networks는 Image Classification에서 획기적인 흐름을 이끌어왔다.
- Deep Networks low/mid/high 단계의 Feature 통합한다. 그리고 Multi-Layer의 end-to-end의 Classifier는 널리 퍼지게 하며, 각 Feature의 "Level"은 Layer 깊게 쌓으면서 보강될 수 있다.
- 따라서 Depth는 중요하며, 결과를 이끈다.

![1.JPG](Deep%20Residual%20Learning%20for%20Image%20Recognition%207354b7eeff774dcf9f7f9452d889cf1a/1.jpg)

- Depth의 중요성에 입각하여, "Network에 Layer 추가하는 방향이 더 좋은가? 에 대한 질문이 따라오게 된다.
- 이 질문에 답하기에 가장 큰 장애물은 `Vanishing/Exploding Gradients` 문제이다.
- 이 문제에 대해 `normalized initialization` 및 `intermediate normalization layers` 10개 이상의 Layer 수렴하게 만들었다.
- 그러나 Network가 깊어지면 정확도 하락 문제가 발생하였다. 이 하락은 Overfitting 인해 발생된 것은 아니다. 즉, Deep Model이 Shallower Model에 비해 높은 Training Error 발생시켰다.
- 해당 논문에서 Layer 직접적으로 쌓지 않고, `Residual Mapping` 이용한다.

$F(x) := H(x) - x$

![2.JPG](Deep%20Residual%20Learning%20for%20Image%20Recognition%207354b7eeff774dcf9f7f9452d889cf1a/2.jpg)

- 해당 공식의 경우 `Shortcut Conntection` 이해할 수 있다. 이는 하나 또는 여러 Layer 건너 뛴다.
- 저자들은 `Identiy Mapping` 이용하여 수행하였고, 이 output은 쌓여진 Layer에 더해진다. `Identiy Shortcut Connection` 추가 Parameter가 필요하지 않고, Computation 또한 복잡하지 않다.

<aside>
📌 **[ResNet 실험 결과]**
1. Depth가 증가할 수록 다른 Network는 Training Error가 높은 반면 Deep Residual Nets은 그렇지 않다.
2. Depth가 증가 해도 이전 Network에 비해 더 높은 정확도를 얻을 수 있다.

</aside>

> **Deep Residual Learning**
> 

**Residual Learning**

- $H(x)$에서 x는 이러한 Layer의 Input 지칭한다. 따라서 H(x)에 유사하게 만드는 대신 F(x) := H(x) - x에 근접 시키도록 한다. 따라서 F(x) + x가 된다. 두 형태 모두 추정할 수 있어야 하지만, 학습의 용이성은 다를 수 있다.
- 만약 추가 Layer가 `Identiy Mappings` 이용하여 설계된다면, Deep Model은 Shallower Model 보다  Training Error가 낮을 것이다.
- `Optimal Function` 가 Zero Mapping 보다 Identity Mapping에 가까울 경우, Solver는 새로운 함수를 학습하는 것보다 Identity Mapping 수정하는 것이 더 쉬워야 한다.

 **Identity Mapping by Shorcuts**

- $y = F(x, [w^i]) + x$ Building Block 다음과 같은 수식으로 정의할 수 있다. 해당 수식에서 x와 y는 input 및 output 지칭하고, F(x)는 Residual Mapping 학습 수식을 의미한다.
- $F = W2∂(W1x)$ ∂는 `ReLU` 의미하고, `bias`는 생략된다.
- F + x 연산은 `Element - Wise Addition` 진행된다.
- F와 x의 `Dimensions`이 다른 경우 `Linear Projection` 통해 Shortcut Connection 진행한다.

<aside>
📌 Element-wise Addition 2개의 Feature map에서 Channel by Channel 진행

</aside>

**Network Architectures**

![4.JPG](Deep%20Residual%20Learning%20for%20Image%20Recognition%207354b7eeff774dcf9f7f9452d889cf1a/4.jpg)

![3.JPG](Deep%20Residual%20Learning%20for%20Image%20Recognition%207354b7eeff774dcf9f7f9452d889cf1a/3.jpg)

- ResNet (Residual Network) 구성하면서 2가지 Option 고려한다.
1. Shortcut은 zero padded 하여 차원을 증가 시키며 수행된다. (Parameter 증가 없음)
2. Dimension Maching 하기 위해 Projection 사용한다. (1 x 1 convolution)

<aside>
📌 Skip Connection 구현할 경우, Stride = 2에 대한 부분에 대해 명시가 되어 있지 않다.
또한 input → 1 x 1 Convolution 진행할 때, Stride = 2 적용하자.

</aside>

> **Experiments**
> 

**Deeper Bottlenect Architectures**

![5.JPG](Deep%20Residual%20Learning%20for%20Image%20Recognition%207354b7eeff774dcf9f7f9452d889cf1a/5.jpg)

- Training-Time 인하여 Building Block → `BottleNeck` 디자인하였다.
- 해당 `1 x 1 Convolution` 차원을 줄이고 늘리는 기능만 담당한다. 일반 3 x 3 연산과 동일한 Complexity 갖는다.
- BottleNect 이용하여 Identity Shortcut 효과적인 모델을 생성할 수 있다.