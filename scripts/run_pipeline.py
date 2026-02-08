import argparse
from pipeline.orchestrator import PipelineOperator

def main():
    parser = argparse.ArgumentParser(description="AF3 Pipeline Operator")
    parser.add_argument("--config", type=str, default="configs/smoke_test.yaml")
    parser.add_argument("--output", type=str, default="./runs")
    args = parser.parse_args()

    op = PipelineOperator(config_path=args.config, output_base=args.output)
    op.execute()

if __name__ == "__main__":
    main()
