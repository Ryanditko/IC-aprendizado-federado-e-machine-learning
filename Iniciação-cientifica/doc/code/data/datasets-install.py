import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import os

def download_datasets():
    """Baixa e prepara os datasets necessários"""
    
    # Criar diretórios se não existirem
    penguin_dir = r"c:\Users\Administrador\Iniciação-Cientifica\doc\code\penguin-dataset"
    iris_dir = r"c:\Users\Administrador\Iniciação-Cientifica\doc\code\iris-dataset"
    
    os.makedirs(penguin_dir, exist_ok=True)
    os.makedirs(iris_dir, exist_ok=True)
    
    print("📥 Baixando datasets...")
    
    try:
        # 1. Dataset Penguin via seaborn
        print("🐧 Baixando Penguin dataset...")
        penguins = sns.load_dataset("penguins")
        penguin_path = os.path.join(penguin_dir, "penguins.csv")
        penguins.to_csv(penguin_path, index=False)
        print(f"✅ Penguin dataset salvo em: {penguin_path}")
        print(f"   Shape: {penguins.shape}")
        print(f"   Colunas: {list(penguins.columns)}")
        
        # 2. Dataset Iris via sklearn
        print("\n🌸 Baixando Iris dataset...")
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['species'] = iris.target_names[iris.target]
        iris_path = os.path.join(iris_dir, "iris.csv")
        iris_df.to_csv(iris_path, index=False)
        print(f"✅ Iris dataset salvo em: {iris_path}")
        print(f"   Shape: {iris_df.shape}")
        print(f"   Colunas: {list(iris_df.columns)}")
        
        print("\n🎉 Todos os datasets foram baixados com sucesso!")
        print("\n📊 Resumo dos datasets:")
        print(f"Penguin dataset: {penguins.shape[0]} linhas, {penguins.shape[1]} colunas")
        print(f"Iris dataset: {iris_df.shape[0]} linhas, {iris_df.shape[1]} colunas")
        
        # Verificar valores faltantes no penguin
        missing_penguin = penguins.isnull().sum().sum()
        if missing_penguin > 0:
            print(f"\n⚠️ Penguin dataset tem {missing_penguin} valores faltantes (será tratado no código)")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao baixar datasets: {str(e)}")
        print("Certifique-se de ter as bibliotecas instaladas:")
        print("pip install pandas seaborn scikit-learn")
        return False

if __name__ == "__main__":
    success = download_datasets()
    if success:
        print("\n🚀 Agora você pode executar os experimentos:")
        print("cd doc\\code\\penguin-dataset && python modelagem.py")
        print("cd doc\\code\\iris-dataset && python modelagem.py")



