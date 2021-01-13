% 데이터 로드
% 한 사람의 그래프 보려면 다른 사람들꺼 주석처리 해야함
data_BSW = load('BSW_200629.mat')
data_EHS = load('EHS_200629.mat')
data_HSW = load('HSW_200629.mat')

first_measure = 65535 % 처음 109초 끝나는 곳의 행

% BSW 첫 측정 그래프. x축은 초, y축은 Fp1(파랑), Fp2(주황)
plot(data_BSW.data(1:first_measure,1),data_BSW.data(1:first_measure,2));
hold on
plot(data_BSW.data(1:first_measure,1),data_BSW.data(1:first_measure,3));
hold off

% EHS 첫 측정 그래프. x축은 초, y축은 Fp1(파랑), Fp2(주황)
plot(data_EHS.data(1:first_measure,1),data_EHS.data(1:first_measure,2));
hold on
plot(data_EHS.data(1:first_measure,1),data_EHS.data(1:first_measure,3));
hold off

% HSW 첫 측정 그래프. x축은 초, y축은 Fp1(파랑), Fp2(주황)
plot(data_HSW.data(1:first_measure,1),data_HSW.data(1:first_measure,2));
hold on
plot(data_HSW.data(1:first_measure,1),data_HSW.data(1:first_measure,3));
hold off

