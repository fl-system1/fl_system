package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	pb "github.com/fl-system1/fl_system/server/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type ClientStatus struct {
	LastUpdate   *pb.ClientUpdate
	LastSeenUnix int64
	HasSentModel bool
}

type server struct {
	pb.UnimplementedFederatedLoggerServer

	mu sync.Mutex

	clients map[string]*ClientStatus

	receivedModels map[string]map[string][]float32
	globalModel    map[string][]float32

	modelUpdates map[string]*pb.ModelUpdate

	expectedClients int
	round           int
}

// added offload_worker connection
var workerClient pb.FederatedLoggerClient

func connectWorker() {
	conn, err := grpc.Dial(
		"offload_worker:50052",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		log.Fatalf("Failed to connect to worker: %v", err)
	}

	workerClient = pb.NewFederatedLoggerClient(conn)
}

func newServer() *server {
	return &server{
		modelUpdates: make(map[string]*pb.ModelUpdate),
	}
}

func NewServer(expected int) *server {
	return &server{
		clients:         make(map[string]*ClientStatus),
		receivedModels:  make(map[string]map[string][]float32),
		globalModel:     make(map[string][]float32),
		round:           0,
		expectedClients: expected,
		modelUpdates:    make(map[string]*pb.ModelUpdate),
	}
}

// SendClientUpdate logs metrics and updates client status
func (s *server) SendClientUpdate(ctx context.Context, update *pb.ClientUpdate) (*pb.Ack, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	cs, ok := s.clients[update.ClientId]
	if !ok {
		cs = &ClientStatus{}
	}

	cs.LastUpdate = update
	cs.LastSeenUnix = time.Now().Unix()
	s.clients[update.ClientId] = cs

	log.Printf("[ClientUpdate] %s cpu=%.2f mem=%.2f%% offloaded=%t\n",
		update.ClientId, update.CpuPercent, update.MemoryPercent, update.Offloaded)

	return &pb.Ack{Message: "Client update received"}, nil
}

// SendModelUpdate receives serialized weights
func (s *server) SendModelUpdate(ctx context.Context, mu *pb.ModelUpdate) (*pb.Ack, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	log.Printf("[ModelUpdate] recv from %s\n", mu.ClientId)

	m := make(map[string][]float32)

	for k, w := range mu.Weights {
		arr := make([]float32, len(w.Values))
		for i, v := range w.Values {
			arr[i] = v
		}
		m[k] = arr
	}

	s.receivedModels[mu.ClientId] = m

	cs, ok := s.clients[mu.ClientId]
	if !ok {
		cs = &ClientStatus{}
	}

	cs.HasSentModel = true
	cs.LastSeenUnix = time.Now().Unix()
	s.clients[mu.ClientId] = cs

	if len(s.receivedModels) >= s.expectedClients {
		log.Println("[Aggregator] Expected clients reached — performing FedAvg")
		go s.performFedAvg()
	}

	return &pb.Ack{Message: "Model update received"}, nil
}

func (s *server) GetGlobalModel(ctx context.Context, req *pb.GlobalModelRequest) (*pb.GlobalModel, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	weights := make(map[string]*pb.Weights)

	for k, arr := range s.globalModel {
		w := &pb.Weights{}
		for _, v := range arr {
			w.Values = append(w.Values, v)
		}
		weights[k] = w
	}

	return &pb.GlobalModel{
		Round:   int64(s.round),
		Weights: weights,
	}, nil
}

func (s *server) performFedAvg() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.receivedModels) == 0 {
		return
	}

	var first map[string][]float32
	for _, v := range s.receivedModels {
		first = v
		break
	}

	avg := make(map[string][]float32)

	for k, arr := range first {
		avg[k] = make([]float32, len(arr))
	}

	numClients := float32(len(s.receivedModels))

	for _, clientModel := range s.receivedModels {
		for k, arr := range clientModel {

			if _, ok := avg[k]; !ok {
				continue
			}

			for i := range arr {
				avg[k][i] += arr[i]
			}
		}
	}

	for k := range avg {
		for i := range avg[k] {
			avg[k][i] = avg[k][i] / numClients
		}
	}

	s.globalModel = avg
	s.round += 1

	s.receivedModels = make(map[string]map[string][]float32)

	log.Printf("[Aggregator] FedAvg complete. New round: %d\n", s.round)
}

// REST handlers for frontend
func (s *server) restClientsHandler(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make(map[string]*ClientStatus)

	for k, v := range s.clients {
		out[k] = v
	}

	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")

	json.NewEncoder(w).Encode(out)
}

func (s *server) restModelHandler(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()

	modelInfo := map[string]interface{}{
		"round":     len(s.globalModel) == 0 && s.round == 0 && s.expectedClients != 0,
		"round_int": s.round,
		"layers":    len(s.globalModel),
	}

	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")

	json.NewEncoder(w).Encode(modelInfo)
}

func (s *server) OffloadTrain(ctx context.Context, req *pb.OffloadRequest) (*pb.ModelUpdate, error) {

	log.Println("[SERVER] Forwarding job to offload worker")

	response, err := workerClient.OffloadTrain(ctx, req)
	if err != nil {
		log.Println("Worker error:", err)
		return nil, err
	}

	log.Println("[SERVER] Received trained model from worker")

	return response, nil
}

func main() {

	connectWorker()

	expectedStr := os.Getenv("EXPECTED_CLIENTS")
	expected := 3

	if expectedStr != "" {
		if v, err := strconv.Atoi(expectedStr); err == nil {
			expected = v
		}
	}

	s := NewServer(expected)

	http.HandleFunc("/status", s.restClientsHandler)
	http.HandleFunc("/model", s.restModelHandler)

	go func() {
		log.Println("REST server listening on :8080")
		if err := http.ListenAndServe(":8080", nil); err != nil {
			log.Fatalf("rest server err: %v", err)
		}
	}()

	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()

	pb.RegisterFederatedLoggerServer(grpcServer, s)

	fmt.Println("gRPC Server listening on :50051")

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
