from methods.process import Experiment


class Evaluate(Experiment):
    def __init__(self, pipeline_model: str):
        super().__init__()
        self.pipeline_model = pipeline_model

    def shapley_additive_explanations(self):
        import shap

        model = None
        features = None

        pipeline = self.perform_single_experiment(pipeline_model=self.pipeline_model, return_pipe=True)
        if self.pipeline_model == 'random-forest':
            model = pipeline.named_steps['classifier']
            features = pipeline.named_steps['vectorizer']
            features_fitted = features.transform(self.X_test)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_fitted)
            shap.summary_plot(shap_values, features_fitted, feature_names=features.get_feature_names_out())



if __name__ == '__main__':
    evaluator = Evaluate(pipeline_model='random-forest')
    evaluator.shapley_additive_explanations()