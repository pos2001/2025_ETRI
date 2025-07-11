# qml: pennylane 라이브러리를 qml이라는 별칭으로 가져옵니다
import pennylane as qml
import pennylane.numpy as np

# braket.jobs.metrics: Amazon Braket은 하이브리드 양자 작업의 성능을 추적하고 모니터링하기 위한 메트릭 로깅 기능을 제공
from braket.jobs.metrics import log_metric

# run_qaoa 함수 정의
# p: 회로의 깊이, 레이어의 수, n_iterations: QAOA 파라미터를 최적화하기 위해 수행하는 반복 횟수, step_size: 경사 하강법(Gradient Descent) 최적화 과정에서 파라미터를 업데이트하는 크기를 결정하는 학습률(learning rate)
def run_qaoa(graph, device, p, n_iterations, step_size):
    
    # num_nodes: 입력 그래프의 노드 개수를 계산하여 저장, 이 값은 QAOA 회로에서 필요한 큐비트 수를 결정하는데 사용됨
    num_nodes = len(graph.nodes)

    # 해밀토니안 정의
    # qml.qaoa:  pennylane 라이브러리의 QAOA 모듈,
    # 아래 maxcut 함수는 입력으로 받은 graph(그래프 데이터)를 기반으로 맥스 컷 문제의 코스트 해밀토니안과 믹서 해밀토니안을 생성하여 리턴
    # cost_h: 최적화하려는 문제를 표현하는 비용 해밀토니안
    # mixer_h: 상태 공간 탐색을 위한 믹서 해밀토니안
    cost_h, mixer_h = qml.qaoa.maxcut(graph)
    
    print(f"Cost hamiltonian: {cost_h}")
    print(f"Mixer hamiltonian: {mixer_h}")

    # QAOA 레이어 구현
    # 이 함수는 QAOA의 단일 레이어를 정의, gamma와 beta는 최적화될 파라미터, 감마는 비용, 베타는 공간
    # 두 개의 주요 연산을 순차적으로 적용
    # 전체 QAOA 회로에서는 이 qaoa_layer 함수가 여러 번 반복 적용되며, 
    # 각 반복마다 다른 gamma와 beta 값을 사용합니다. 이렇게 함으로써 알고리즘은 문제의 해 공간을 효과적으로 탐색할 수 있게 됩니다
    def qaoa_layer(gamma, beta):
        
        #비용 해밀토니안(cost_h)에 대한 시간 발전 연산자 적용
        qml.qaoa.cost_layer(gamma, cost_h)
        
        qml.qaoa.mixer_layer(beta, mixer_h)

    # 전체 회로 구현
    # circuit: qaoa 양자 회로 정의, 페니레인 라이브러리 사용
    # 먼저 모든 큐비트에 Hadamard 게이트를 적용합니다. 그 다음, qml.layer 함수를 사용하여 QAOA 레이어를 반복적으로 적용  
    # 직전에 정의한 QAOA 레이어를 p번 반복 적용합니다. 각 레이어는 비용 Hamiltonian과 믹서 Hamiltonian을 번갈아 적용
    # parameters는 QAOA 알고리즘의 변분 파라미터들을 포함하는 배열
    def circuit(parameters):
        # 이 루프는 회로의 모든 큐비트에 대해 반복
        for i in range(num_nodes):
            # 각 큐비트에 Hadamard 게이트를 적용
            qml.Hadamard(wires=i)
        
        # qml.layer 함수를 사용하여 QAOA 층을 지정된 만큼 반복적으로 적용
        # p는 QAOA 레이어의 반복 횟수,parameters[0]와 parameters[1]는 각각 감마(gamma)와 베타(beta) 파라미터 배열
        qml.layer(qaoa_layer, p, parameters[0], parameters[1])
        
    # 이 회로는 QAOA 알고리즘의 핵심 부분으로, 최적의 파라미터를 찾아 Max-Cut 문제의 근사해를 구하는 데 사용

    
    # @qml.qnode(device): PennyLane의 데코레이터로, 양자 회로를 지정된 디바이스에서 실행하도록 지정합니다
    @qml.qnode(device)
    
    # 비용 함수 정의
    # 이 함수는 입력 매개변수(parameters)를 받아서 비용 함수의 기대값을 계산
    def cost_function(parameters):
        
        # 이 부분은 입력 매개변수를 사용하여 퀀텀 회로를 실행, 여기서 circuit는 이전에 정의된 QAOA 회로
        circuit(parameters)
        
        # 이 부분은 cost_h라는 연산자(operator)의 기대값을 계산하고 그 결과를 반환합니다
        return qml.expval(cost_h)
    # 주어진 파라미터로 양자 회로를 실행 => 비용 해밀토니안의 기대값을 계산 => 이 값을 최소화하는 방향으로 파라미터를 최적화
    
    # QAOA 알고리즘의 초기 파라미터를 생성, 난수를 이용하여 초기 파라미터 생성
    # 난수 생성의 시작점을 설정
    np.random.seed(42)
    
    # np.random.uniform(size=[2, p])는 [0, 1) 범위의 균등 분포에서 난수를 생성하여 2 x p 크기의 배열을 만듭니다
    # 파라미터 배열의 크기는 [2, p]로, 각 레이어마다 고유한 감마(γ)와 베타(β) 값을 가집니다.
    # 0.01 *는 생성된 난수 배열의 값을 0.01배로 스케일링합니다. 이는 초기 파라미터 값을 작게 설정하기 위함입니다
    # np.array(..., requires_grad=True)는 생성된 배열을 PennyLane의 배열 객체로 변환하며, 이 배열은 최적화 과정에서 기울기를 계산할 수 있도록 합니다
    params = np.array(0.01 * np.random.uniform(size=[2, p]), requires_grad=True)
    print(f"Initial parameters: {params}")

    # print("Circuit diagram:")
    # print(qml.draw(cost_function, expansion_strategy="device")(params))

    # Gradient Descent Optimizer를 초기화
    # 경사하강법을 이용하여 파라미터 최적화
    # qml.GradientDescentOptimizer는 PennyLane에서 제공하는 경사 하강법 최적화 알고리즘, 함수의 최소값을 찾기 위한 최적화 알고리즘
    optimizer = qml.GradientDescentOptimizer(stepsize=step_size)

    # 최적화 과정이 시작됨을 출력
    print("Optimization start")
    # n_iterations만큼 반복하며 최적화를 진행
    for iteration in range(n_iterations):
        print(f">>>>>>> - Iteration step {iteration}")

        # Evaluates the cost, then does a gradient step to new params
        # optimizer.step_and_cost() 메서드를 호출하여 비용 함수를 평가하고 파라미터를 업데이트
        # 새로운 파라미터와 업데이트 전의 비용 기대값을 반환
        # cost_function: 앞서 정의
        params, cost_before = optimizer.step_and_cost(cost_function, params)

        # Track the cost as a metric
        log_metric(
            metric_name="cost",
            value=float(cost_before),
            iteration_number=iteration,
        )
    # 각 반복에서 비용 함수를 평가하고, 파라미터를 업데이트하며, 현재의 비용을 로깅합니다. 이 과정을 통해 알고리즘은 점진적으로 더 나은 해를 찾아갑니다. 메트릭 로깅을 통해 최적화 과정의 진행 상황을 추적하고 분석
        
        
        
    print("Optimization completed")
    # 최적화가 완료된 후의 최종 파라미터 값을 optimized_parameters 변수에 저장
    optimized_parameters = params
    # 최적화된 파라미터를 사용하여 비용 함수를 평가하고, 그 결과를 final_cost 변수에 저장
    final_cost = float(cost_function(optimized_parameters))
    print(f"Cost after optimization: {final_cost}")
    print(f"Parameters after optimization: {optimized_parameters}")

    
    # 이 코드는 양자 머신러닝 라이브러리인 PennyLane를 사용하여 양자 회로에서 비트 문자열의 확률을 계산합니다. 
    # Sample bitstring probabilities from optimized circuit
    @qml.qnode(device)
    # 이 함수는 입력으로 parameters를 받습니다. 이 함수는 주어진 파라미터로 양자 회로를 실행하고 측정 결과의 확률 분포를 반환
    def sampled_probs(parameters):
        # 실제 양자 회로가 실행됩니다. circuit은 다른 곳에서 정의된 함수로, 입력 매개변수를 사용하여 양자 회로를 구성
        circuit(parameters)
        # qml.probs() 이 함수는 현재 양자 상태의 확률 분포를 계산합니다
        return qml.probs()
    # 이 줄은 모든 가능한 비트열을 생성합니다. num_nodes는 양자 회로에 사용된 큐비트의 수입니다. 예를 들어, num_nodes가 3이면 bit_strings는 ['000', '001', '010', '011', '100', '101', '110', '111']이 됩니다.
    bit_strings = ["{0:{fill}{length}b}".format(i, fill='0', length=num_nodes) for i in range(2 ** num_nodes)]
    probs = dict(zip(bit_strings, sampled_probs(optimized_parameters)))

    return optimized_parameters, probs



# 이 코드는 양자 컴퓨팅 디바이스를 초기화하는 함수입니다:
# n_wires: 양자 회로에서 사용할 큐비트 수입니다. 기본값은 1
# max_parallel: 최대 병렬 실행 수를 설정합니다. 기본값은 1
# n_shots: 측정 샷 수입니다. 기본값은 None으로, 이는 기본 샷 수를 사용함을 의미
def get_device_instance(device_arn, n_wires=1, n_shots=None, max_parallel=1):
    device_prefix = device_arn.split(":")[0]
    # 로컬 디바이스인 경우
    if device_prefix == "local":
        prefix, device_name = device_arn.split("/")
        device = qml.device(
            device_name,
            wires=n_wires,
            shots=n_shots
        )
    # AWS Braket 디바이스인 경우
    else:
        device = qml.device(
            "braket.aws.qubit",
            device_arn=device_arn,
            wires=n_wires,
            shots=n_shots,
            parallel=True,
            max_parallel=max_parallel,
        )
    print(f"Device: {device.name}")
    print(f"Qubits: {n_wires}")
    print(f"Shots: {n_shots}")
    return device

def run_qaoa(graph, device, p, n_iterations, step_size):
    
    # num_nodes: 입력 그래프의 노드 개수를 계산하여 저장, 이 값은 QAOA 회로에서 필요한 큐비트 수를 결정하는데 사용됨
    num_nodes = len(graph.nodes)

    # 해밀토니안 정의
    # qml.qaoa:  pennylane 라이브러리의 QAOA 모듈,
    # 아래 maxcut 함수는 입력으로 받은 graph(그래프 데이터)를 기반으로 최대 차단 문제의 코스트 해밀토니안과 믹서 해밀토니안을 생성
    # cost_h: 최적화하려는 문제를 표현하는 비용 해밀토니안
    # mixer_h: 상태 공간 탐색을 위한 믹서 해밀토니안
    cost_h, mixer_h = qml.qaoa.maxcut(graph)
    
    print(f"Cost hamiltonian: {cost_h}")
    print(f"Mixer hamiltonian: {mixer_h}")

    # QAOA 레이어 구현
    # 이 함수는 QAOA의 단일 레이어를 정의, gamma와 beta는 최적화될 파라미터, 감마는 비용, 베타는 공간
    # 두 개의 주요 연산을 순차적으로 적용
    # 전체 QAOA 회로에서는 이 qaoa_layer 함수가 여러 번 반복 적용되며, 
    # 각 반복마다 다른 gamma와 beta 값을 사용합니다. 이렇게 함으로써 알고리즘은 문제의 해 공간을 효과적으로 탐색할 수 있게 됩니다
    def qaoa_layer(gamma, beta):
        #비용 해밀토니안(cost_h)에 대한 시간 발전 연산자 적용
        qml.qaoa.cost_layer(gamma, cost_h)
        
        qml.qaoa.mixer_layer(beta, mixer_h)

    # 전체 회로 구현
    # circuit: qaoa 양자 회로 정의, 페니레인 라이브러리 사용
    # 먼저 모든 큐비트에 Hadamard 게이트를 적용합니다. 그 다음, qml.layer 함수를 사용하여 QAOA 레이어를 반복적으로 적용  
    # 직전에 정의한 QAOA 레이어를 p번 반복 적용합니다. 각 레이어는 비용 Hamiltonian과 믹서 Hamiltonian을 번갈아 적용
    # parameters는 QAOA 알고리즘의 변분 파라미터들을 포함하는 배열
    def circuit(parameters):
        # 이 루프는 회로의 모든 큐비트에 대해 반복
        for i in range(num_nodes):
            # 각 큐비트에 Hadamard 게이트를 적용
            qml.Hadamard(wires=i)
        # qml.layer 함수를 사용하여 QAOA 층을 지정된 만큼 반복적으로 적용
        # p는 QAOA 레이어의 반복 횟수,parameters[0]와 parameters[1]는 각각 감마(gamma)와 베타(beta) 파라미터 배열
        qml.layer(qaoa_layer, p, parameters[0], parameters[1])
        
    # 이 회로는 QAOA 알고리즘의 핵심 부분으로, 최적의 파라미터를 찾아 Max-Cut 문제의 근사해를 구하는 데 사용

    
    
    # @qml.qnode(device): PennyLane의 데코레이터로, 양자 회로를 지정된 디바이스에서 실행하도록 지정합니다
    @qml.qnode(device)
    # 비용 함수 정의
    # 이 함수는 입력 매개변수(parameters)를 받아서 비용 함수의 기대값을 계산
    def cost_function(parameters):
        
        # 이 부분은 입력 매개변수를 사용하여 퀀텀 회로를 실행, 여기서 circuit는 이전에 정의된 QAOA 회로
        circuit(parameters)
        
        # 이 부분은 cost_h라는 연산자(operator)의 기대값을 계산하고 그 결과를 반환합니다
        return qml.expval(cost_h)
    # 주어진 파라미터로 양자 회로를 실행 => 비용 해밀토니안의 기대값을 계산 => 이 값을 최소화하는 방향으로 파라미터를 최적화
    
    # QAOA 알고리즘의 초기 파라미터를 생성, 난수를 이용하여 초기 파라미터 생성
    # 난수 생성의 시작점을 설정
    np.random.seed(42)
    
    # np.random.uniform(size=[2, p])는 [0, 1) 범위의 균등 분포에서 난수를 생성하여 2 x p 크기의 배열을 만듭니다
    # 파라미터 배열의 크기는 [2, p]로, 각 레이어마다 고유한 감마(γ)와 베타(β) 값을 가집니다.
    # 0.01 *는 생성된 난수 배열의 값을 0.01배로 스케일링합니다. 이는 초기 파라미터 값을 작게 설정하기 위함입니다
    # np.array(..., requires_grad=True)는 생성된 배열을 PennyLane의 배열 객체로 변환하며, 이 배열은 최적화 과정에서 기울기를 계산할 수 있도록 합니다
    params = np.array(0.01 * np.random.uniform(size=[2, p]), requires_grad=True)
    print(f"Initial parameters: {params}")

    # print("Circuit diagram:")
    # print(qml.draw(cost_function, expansion_strategy="device")(params))

    # Gradient Descent Optimizer를 초기화
    # 경사하강법을 이용하여 파라미터 최적화
    # qml.GradientDescentOptimizer는 PennyLane에서 제공하는 경사 하강법 최적화 알고리즘
    optimizer = qml.GradientDescentOptimizer(stepsize=step_size)

    # 최적화 과정이 시작됨을 출력
    print("Optimization start")
    # n_iterations만큼 반복하며 최적화를 진행
    for iteration in range(n_iterations):
        print(f">>>>>>> - Iteration step {iteration}")

        # Evaluates the cost, then does a gradient step to new params
        # optimizer.step_and_cost() 메서드를 호출하여 비용 함수를 평가하고 파라미터를 업데이트
        # 새로운 파라미터와 업데이트 전의 비용 기대값을 반환
        # cost_function: 앞서 정의
        params, cost_before = optimizer.step_and_cost(cost_function, params)

        # Track the cost as a metric
        log_metric(
            metric_name="cost",
            value=float(cost_before),
            iteration_number=iteration,
        )
    # 각 반복에서 비용 함수를 평가하고, 파라미터를 업데이트하며, 현재의 비용을 로깅합니다. 이 과정을 통해 알고리즘은 점진적으로 더 나은 해를 찾아갑니다. 메트릭 로깅을 통해 최적화 과정의 진행 상황을 추적하고 분석
        
        
        
    print("Optimization completed")
    # 최적화가 완료된 후의 최종 파라미터 값을 optimized_parameters 변수에 저장
    optimized_parameters = params
    # 최적화된 파라미터를 사용하여 비용 함수를 평가하고, 그 결과를 final_cost 변수에 저장
    final_cost = float(cost_function(optimized_parameters))
    print(f"Cost after optimization: {final_cost}")
    print(f"Parameters after optimization: {optimized_parameters}")

    
    # 이 코드는 양자 머신러닝 라이브러리인 PennyLane를 사용하여 양자 회로에서 비트 문자열의 확률을 계산합니다. 
    # Sample bitstring probabilities from optimized circuit
    @qml.qnode(device)
    # 이 함수는 입력으로 parameters를 받습니다. 이 함수는 주어진 파라미터로 양자 회로를 실행하고 측정 결과의 확률 분포를 반환
    def sampled_probs(parameters):
        # 실제 양자 회로가 실행됩니다. circuit은 다른 곳에서 정의된 함수로, 입력 매개변수를 사용하여 양자 회로를 구성
        circuit(parameters)
        # qml.probs() 이 함수는 현재 양자 상태의 확률 분포를 계산합니다
        return qml.probs()
    # 이 줄은 모든 가능한 비트열을 생성합니다. num_nodes는 양자 회로에 사용된 큐비트의 수입니다. 예를 들어, num_nodes가 3이면 bit_strings는 ['000', '001', '010', '011', '100', '101', '110', '111']이 됩니다.
    bit_strings = ["{0:{fill}{length}b}".format(i, fill='0', length=num_nodes) for i in range(2 ** num_nodes)]
    probs = dict(zip(bit_strings, sampled_probs(optimized_parameters)))

    return optimized_parameters, probs



# 이 코드는 양자 컴퓨팅 디바이스를 초기화하는 함수입니다:
# n_wires: 양자 회로에서 사용할 큐비트 수입니다. 기본값은 1
# max_parallel: 최대 병렬 실행 수를 설정합니다. 기본값은 1
# n_shots: 측정 샷 수입니다. 기본값은 None으로, 이는 기본 샷 수를 사용함을 의미
def get_device_instance(device_arn, n_wires=1, n_shots=None, max_parallel=1):
    device_prefix = device_arn.split(":")[0]
    
    
    # 로컬 디바이스인 경우
    # 로컬 디바이스는 원격 양자 하드웨어나 클라우드 기반 시뮬레이터가 아닌, 사용자의 로컬 컴퓨터에서 실행되는 양자 시뮬레이터를 의미합니다.
    # 예를 들어, "local:pennylane/lightning.qubit"이라는 디바이스 ARN을 사용하면
    if device_prefix == "local":
        prefix, device_name = device_arn.split("/")
        device = qml.device(
            device_name,
            wires=n_wires,
            shots=n_shots
        )
    # AWS Braket 디바이스인 경우
    else:
        device = qml.device(
            "braket.aws.qubit",
            device_arn=device_arn,
            wires=n_wires,
            shots=n_shots,
            parallel=True,
            max_parallel=max_parallel,
        )
    print(f"Device: {device.name}")
    print(f"Qubits: {n_wires}")
    print(f"Shots: {n_shots}")
    return device
