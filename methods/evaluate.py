import pandas as pd

from methods.process import Experiment


class Evaluate(Experiment):
    def __init__(self, pipeline_model: str):
        super().__init__()
        self.pipeline_model = pipeline_model

    def coefficient_weights(self):

        pipeline = self.perform_single_experiment(pipeline_model=self.pipeline_model, return_pipe=True)

        logreg_coef = pipeline.get_params()['classifier'].coef_[0]
        print(len(logreg_coef))
        tfidf_mapping = pipeline.get_params()['vectorizer'].vocabulary_
        tfidf_map = [(x, tfidf_mapping[x]) for x in tfidf_mapping]
        print(tfidf_map[:10])
        vocab_tfidf = [x[0] for x in tfidf_map]

        tfidf_map.sort(key=lambda x: x[1])
        coef_df = pd.DataFrame(list(zip(vocab_tfidf, logreg_coef)), columns=['word', 'coef'])
        coef_df.sort_values(by='coef', ascending=False).head()
        print(coef_df)


if __name__ == '__main__':
    evaluator = Evaluate(pipeline_model='logistic')

