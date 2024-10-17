import sdv.evaluation.single_table as sdv_st
from sdv.metadata import SingleTableMetadata
import pandas as pd


def svd_evaluation(real_data, synthetic_data, savepath, show=False):
    # svd evaluation
    metadata = SingleTableMetadata()  # build a metadata for evaluation (from SDV)
    metadata.detect_from_dataframe(data=real_data)
    quality_report = sdv_st.evaluate_quality(  # generate quality report
        real_data,
        synthetic_data,
        metadata=metadata
    )
    print(quality_report)  # print quality report

    properties_dataframe = quality_report.get_properties()
    Column_Shapes = properties_dataframe.iloc[0, 1]
    Column_Pair_Trends = properties_dataframe.iloc[1, 1]
    print('Column Shapes:', Column_Shapes)
    print('Column Pair Trends:', Column_Pair_Trends)
    properties_dataframe.to_csv(savepath + '_properties_dataframe.csv', index=False)

    if show:
        # plot Column Shapes -> referred to the "Fidelity Column" in the paper
        fig = quality_report.get_visualization(property_name='Column Shapes')
        fig.write_html(savepath + '_Column_Shapes.html')
        fig.show()
        # plot Column Pair Trends -> referred to the "Fidelity Row" in the paper
        fig = quality_report.get_visualization(property_name='Column Pair Trends')
        fig.write_html(savepath + '_Column_Pair_Trends.html')
        fig.show()
    return Column_Shapes, Column_Pair_Trends


if __name__ == '__main__':
    # 假设我们有一些示例的真实数据和合成数据
    # 这里我们只是简单地创建两个包含随机数的DataFrame作为示例
    real_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [2, 3, 4, 5, 6]
    })

    synthetic_data = pd.DataFrame({
        'A': [1.1, 1.9, 2.8, 4.5, 4.9],
        'B': [5.2, 4.3, 3.4, 2.5, 1.7],
        'C': [2.1, 3.4, 4.5, 5.6, 6.7]
    })

    # 定义一个保存路径
    savepath = "./"

    # 调用 svd_evaluation 函数
    svd_evaluation(real_data, synthetic_data, savepath)
