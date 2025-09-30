from config.name.data_analysis.AnalysisModelEnName import AnalysisModelEnName
from entity.deep_learning_model.AttentionVAE import AttentionVAE
from entity.deep_learning_model.AttentionLSTM import AttentionLSTM
from entity.deep_learning_model.AttentionSTAE import AttentionSTAE
from entity.deep_learning_model.AutoEncoder import AutoEncoder
from entity.deep_learning_model.BayesianNN import BayesianNN
from entity.deep_learning_model.BiScaleWaveNet import BiScaleWaveNet
from entity.machine_learning_model.BayesianRegression import BayesianRegression
from entity.machine_learning_model.DecisionTree import DecisionTree
from entity.deep_learning_model.LSTM import LSTM
from entity.machine_learning_model.KernelBayesianRegression import KernelBayesianRegression
from entity.machine_learning_model.LightGBM import LightGBM
from entity.machine_learning_model.Logistic import Logistic
from entity.deep_learning_model.MultiHorizonQuantileRNN import MultiHorizonQuantileRNN
from entity.deep_learning_model.RAE import RAE
from entity.deep_learning_model.RVAE import RVAE
from entity.machine_learning_model.RandomForest import RandomForest
from entity.machine_learning_model.SVM import SVM
from entity.deep_learning_model.Transformer import Transformer
from entity.deep_learning_model.VAE import VAE
from entity.deep_learning_model.RegressionWaveNet import RegressionWaveNet


class ModelFactory:
    @classmethod
    def get_model(cls, model_name, device, hyper_params: dict):
        model_mapping = {
            AnalysisModelEnName.BiScaleWaveNet: lambda: BiScaleWaveNet(
                input_dim=hyper_params["input_dim"],
                output_dim=hyper_params["output_dim"]
            ).to(device),
            AnalysisModelEnName.WaveNet: lambda: RegressionWaveNet(
                input_dim=hyper_params["input_dim"],
                output_dim=hyper_params["output_dim"]
            ).to(device),
            AnalysisModelEnName.LSTM: lambda: LSTM(
                input_dim=hyper_params["input_dim"],
                hidden_dim=hyper_params["hidden_dim"],
                output_dim=hyper_params["output_dim"]
            ).to(device),
            AnalysisModelEnName.Transformer: lambda: Transformer(
                input_dim=hyper_params["input_dim"],
                hidden_dim=hyper_params["hidden_dim"],
                output_dim=hyper_params["output_dim"]
            ).to(device),
            AnalysisModelEnName.AttentionLSTM: lambda: AttentionLSTM(
                input_dim=hyper_params["input_dim"],
                hidden_dim=hyper_params["hidden_dim"],
                output_dim=hyper_params["output_dim"]
            ).to(device),
            AnalysisModelEnName.AutoEncoder: lambda: AutoEncoder(
                hidden_dim1=hyper_params["input_dim"],
                hidden_dim2=hyper_params["hidden_dim2"],
                hidden_dim3=hyper_params["hidden_dim3"],
                hidden_dim4=hyper_params["hidden_dim4"]
            ).to(device),
            AnalysisModelEnName.RAE: lambda: RAE(
                hidden_dim1=hyper_params["input_dim"],
                hidden_dim2=hyper_params["hidden_dim2"],
                hidden_dim3=hyper_params["hidden_dim3"],
                hidden_dim4=hyper_params["hidden_dim4"]
            ).to(device),
            AnalysisModelEnName.VAE: lambda: VAE(
                hidden_dim1=hyper_params["input_dim"],
                hidden_dim2=hyper_params["hidden_dim2"],
                hidden_dim3=hyper_params["hidden_dim3"],
                hidden_dim4=hyper_params["hidden_dim4"]
            ).to(device),
            AnalysisModelEnName.RVAE: lambda: RVAE(
                hidden_dim1=hyper_params["input_dim"],
                hidden_dim2=hyper_params["hidden_dim2"],
                hidden_dim3=hyper_params["hidden_dim3"],
                hidden_dim4=hyper_params["hidden_dim4"]
            ).to(device),
            AnalysisModelEnName.AttentionVAE: lambda: AttentionVAE(
                hidden_dim1=hyper_params["input_dim"],
                hidden_dim2=hyper_params["hidden_dim2"],
                hidden_dim3=hyper_params["hidden_dim3"],
                hidden_dim4=hyper_params["hidden_dim4"]
            ).to(device),
            AnalysisModelEnName.BayesianNN: lambda: BayesianNN(
                hidden_dim1=hyper_params["input_dim"],
                hidden_dim2=hyper_params["hidden_dim2"],
                hidden_dim3=hyper_params["hidden_dim3"],
                hidden_dim4=hyper_params["hidden_dim4"],
                device=device
            ).to(device),
            AnalysisModelEnName.KernelBayesianRegression: lambda: KernelBayesianRegression(
                alpha=hyper_params["alpha"],
                beta=hyper_params["beta"],
                length_scale=hyper_params["length_scale"]
            ),
            AnalysisModelEnName.BayesianRegression: lambda: BayesianRegression(
                alpha=hyper_params["alpha"],
                beta=hyper_params["beta"]
            ),
            AnalysisModelEnName.DecisionTree: lambda: DecisionTree(
            ),
            AnalysisModelEnName.Logistic: lambda: Logistic(
            ),
            AnalysisModelEnName.SVM: lambda: SVM(
            ),
            AnalysisModelEnName.RandomForest: lambda: RandomForest(
            ),
            AnalysisModelEnName.LightGBM: lambda: LightGBM(
            ),
            AnalysisModelEnName.AttentionSTAE: lambda: AttentionSTAE(
                turbine_num=hyper_params["turbine_num"],
                embedding_dim=hyper_params["embedding_dim"],
                input_dim=hyper_params["input_dim"],
                temporal_encoder_hidden_dim=hyper_params["temporal_encoder_hidden_dim"],
                graph_encoder_hidden_dim1=hyper_params["graph_encoder_hidden_dim1"],
                graph_encoder_hidden_dim2=hyper_params["graph_encoder_hidden_dim2"],
                timestep=hyper_params["timestep"],
                dropout=hyper_params["dropout"],
                device=device,
            ).to(device),
        }

        model_constructor = model_mapping.get(model_name, None)
        if model_constructor is None:
            return None

        # 调用lambda模板实现延迟实例化
        return model_constructor()

