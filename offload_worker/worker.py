# offload_worker/worker.py

import os
import sys
import grpc
import time
from concurrent import futures

# load proto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "proto"))

import fl_pb2
import fl_pb2_grpc

from train_worker import train_model, set_model_weights, get_model_weights, create_model


class WorkerService(fl_pb2_grpc.FederatedLoggerServicer):

    def OffloadTrain(self, request, context):

        client_id = request.client_id
        print(f"[WORKER] Offload request from client {client_id}")

        try:
            # 1️⃣ receive weights
            weights = request.weights

            # 2️⃣ create model
            model = create_model()

            # 3️⃣ load received weights
            set_model_weights(model, weights)

            # 4️⃣ train model
            trained_model = train_model(model)

            # 5️⃣ extract updated weights
            updated_weights = get_model_weights(trained_model)

            print("[WORKER] Training complete")

            return fl_pb2.ModelUpdate(
                client_id=client_id,
                weights=updated_weights
            )

        except Exception as e:
            print(f"[WORKER ERROR] {e}")

            return fl_pb2.ModelUpdate(
                client_id=client_id,
                weights={}
            )


def serve():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    fl_pb2_grpc.add_FederatedLoggerServicer_to_server(
        WorkerService(), server
    )

    port = "50052"
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    print(f"[WORKER] Running on port {port}")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()