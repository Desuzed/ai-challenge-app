package agent

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"ai-challenge-app/internal/models"
)

type fakeCompleter struct {
	requests [][]models.ChatMessage
	models   []string
	answer   string
	err      error
	usage    models.ModelUsage
}

func (f *fakeCompleter) CompleteMessagesModel(ctx context.Context, model string, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	f.models = append(f.models, model)
	return f.CompleteMessages(ctx, messages, settings)
}

func TestPersistentAgentRestoresHistoryAfterRestart(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "state", "agent-history.json"))
	firstClient := &fakeCompleter{answer: "Запомнил"}
	firstAgent, err := NewPersistent(firstClient, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := firstAgent.Respond(context.Background(), "session-a", "Мой любимый цвет — зелёный"); err != nil {
		t.Fatal(err)
	}

	secondClient := &fakeCompleter{answer: "Продолжаю"}
	secondAgent, err := NewPersistent(secondClient, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := secondAgent.Respond(context.Background(), "session-a", "Какой мой любимый цвет?"); err != nil {
		t.Fatal(err)
	}
	if got, want := secondClient.requests[0][1:], []models.ChatMessage{
		{Role: "user", Content: "Мой любимый цвет — зелёный"},
		{Role: "assistant", Content: "Запомнил"},
		{Role: "user", Content: "Какой мой любимый цвет?"},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("restored request = %#v, want %#v", got, want)
	}
}

func TestClearRemovesPersistentSession(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "agent-history.json"))
	first, err := NewPersistent(&fakeCompleter{answer: "Ответ"}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.Respond(context.Background(), "session-a", "Сообщение"); err != nil {
		t.Fatal(err)
	}
	if err := first.Clear("session-a"); err != nil {
		t.Fatal(err)
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if got := second.History("session-a"); len(got) != 0 {
		t.Fatalf("history after clear = %#v, want empty", got)
	}
}

func TestTaskStateMachinePausesAndResumesFromSavedStep(t *testing.T) {
	client := &fakeCompleter{answer: "Продолжаю работу"}
	agent := New(client)
	configured, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{
		Goal:           "Подготовить запуск",
		Phases:         []string{"planning", "execution", "validation", "done"},
		CurrentStep:    "Собрать требования",
		ExpectedAction: "Согласовать приоритеты",
	}})
	if err != nil {
		t.Fatal(err)
	}
	if configured.Task.Phase != "planning" || configured.Task.Status != models.TaskActive || configured.Task.PhaseIndex != 0 {
		t.Fatalf("configured task = %#v", configured.Task)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "pause_task"}); err != nil {
		t.Fatal(err)
	}
	paused, err := agent.Respond(context.Background(), "session", "Что делать дальше?")
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 0 || !strings.Contains(paused.Answer, "не было отправлено модели") {
		t.Fatalf("paused response = %#v, calls=%d", paused, len(client.requests))
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "resume_task"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "session", "Продолжаем"); err != nil {
		t.Fatal(err)
	}
	resumedContext := client.requests[0][1].Content
	if !strings.Contains(resumedContext, "Статус: active") || !strings.Contains(resumedContext, "Не повторяй прежнее объяснение") {
		t.Fatalf("resumed task context = %q", resumedContext)
	}
}

func TestPausedTaskDoesNotCallModelUntilResumed(t *testing.T) {
	client := &fakeCompleter{answer: "Модельный ответ"}
	agent := New(client)
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "Собрать MVP", Phases: []string{"planning", "done"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "pause_task"}); err != nil {
		t.Fatal(err)
	}
	paused, err := agent.Respond(context.Background(), "session", "Измени состав экранов")
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 0 || !strings.Contains(paused.Answer, "не было отправлено модели") || len(paused.Messages) != 2 {
		t.Fatalf("paused response = %#v, calls=%d", paused, len(client.requests))
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "resume_task"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "session", "Измени состав экранов"); err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 1 {
		t.Fatalf("calls after resume = %d, want 1", len(client.requests))
	}
}

func TestProjectPlannerBuildsDashboardFromChatMessage(t *testing.T) {
	client := &fakeCompleter{answer: `{"answer":"Черновик ТЗ готов.","plan":{"specification":"MVP: клиентское приложение с каталогом, корзиной и оформлением заказа.","currentStep":"Согласовать состав экранов","expectedAction":"Подтвердить границы MVP","openQuestions":["Нужен ли поиск?"],"decisions":["Делаем только текстовое ТЗ"],"nextSteps":["Описать пользовательские сценарии"]}}`}
	agent := New(client)
	result, err := agent.Respond(context.Background(), "session", "Разработка MVP приложения\n\n- Цель: «Сделать MVP приложения доставки еды».\n- Этапы: `planning → execution → validation → done`.\n- Шаг: «Собрать список ключевых экранов».")
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "Черновик ТЗ готов." || result.Task.Goal != "Сделать MVP приложения доставки еды" || result.Task.Phase != "planning" || result.Task.CurrentStep != "Согласовать состав экранов" || result.Task.Specification == "" || len(result.Task.OpenQuestions) != 1 || len(result.Task.Decisions) != 1 {
		t.Fatalf("planner result = %#v", result)
	}
	if len(client.requests) != 1 || !strings.Contains(client.requests[0][2].Content, "агент-проектировщик") {
		t.Fatalf("planner request = %#v", client.requests)
	}
}

func TestProjectPlannerSalvagesAnswerAndSpecificationFromTruncatedJSON(t *testing.T) {
	answer, plan := applyPlannerCompletion(models.TaskState{Goal: "MVP", CurrentStep: "Исходный шаг"}, `{"answer":"Черновик готов.","plan":{"specification":"Подробное ТЗ для MVP.","currentStep":"Согласовать экраны","expectedAction":"Подтвердить список","openQuestions":["Нужен поиск?"`)
	if answer != "Черновик готов." || plan.Specification != "Подробное ТЗ для MVP." || plan.CurrentStep != "Согласовать экраны" || plan.ExpectedAction != "Подтвердить список" {
		t.Fatalf("salvaged plan = %#v, answer=%q", plan, answer)
	}
}

func TestProjectPlannerStoresFinalArtifactSeparatelyWithoutCompletingTask(t *testing.T) {
	answer, plan := applyPlannerCompletion(models.TaskState{
		Goal:          "MVP доставки",
		Phases:        []string{"planning", "done"},
		PhaseIndex:    1,
		Phase:         "done",
		Status:        models.TaskActive,
		Specification: "Рабочий черновик: согласовать список экранов.",
	}, `{"answer":"Итоговое ТЗ сформировано.","plan":{"artifactTitle":"ТЗ MVP доставки еды","artifactContent":"# ТЗ MVP\n\n## Цель\nСобрать приложение доставки еды.\n\n## Критерии приёмки\nПользователь оформляет заказ."}}`)
	if answer != "Итоговое ТЗ сформировано." || plan.ArtifactTitle != "ТЗ MVP доставки еды" || !strings.Contains(plan.ArtifactContent, "Критерии приёмки") {
		t.Fatalf("artifact = %#v, answer=%q", plan, answer)
	}
	if plan.Specification != "Рабочий черновик: согласовать список экранов." {
		t.Fatalf("working specification was overwritten: %q", plan.Specification)
	}
	if plan.Status == models.TaskDone {
		t.Fatalf("model artifact must not complete task: %#v", plan)
	}
}

func TestProjectPlannerRecognizesSingleLineStarterMessage(t *testing.T) {
	task, ok := taskFromMessage("Разработка MVP приложения - Цель: «Сделать MVP приложения доставки еды». - Этапы: planning → execution → validation → done. - Шаг: «Собрать список ключевых экранов».")
	if !ok || task.Goal != "Сделать MVP приложения доставки еды" || !reflect.DeepEqual(task.Phases, []string{"planning", "execution", "validation", "done"}) || task.CurrentStep != "Собрать список ключевых экранов" {
		t.Fatalf("parsed task = %#v, ok=%v", task, ok)
	}
}

func TestPlanApprovalRefreshesDashboardQuestionsAndNextSteps(t *testing.T) {
	updated := updatePlannerProgress(models.TaskState{
		Goal:           "MVP",
		Phases:         []string{"planning", "execution", "validation", "done"},
		Phase:          "planning",
		PhaseIndex:     0,
		ExpectedAction: "Согласовать состав MVP",
		OpenQuestions:  []string{"Нужен поиск?", "Нужны акции?"},
	}, "Подтверждаю план, всё согласовано.")
	if updated.Phase != "execution" || updated.PhaseIndex != 1 || !updated.PlanApproved || len(updated.OpenQuestions) != 0 || len(updated.Decisions) != 1 || !strings.Contains(updated.CurrentStep, "execution") || !strings.Contains(updated.ExpectedAction, "реализацию") || len(updated.NextSteps) != 1 {
		t.Fatalf("updated dashboard = %#v", updated)
	}
}

func TestProjectPlannerCannotChangePhaseFromModelResponse(t *testing.T) {
	_, plan := applyPlannerCompletion(models.TaskState{Goal: "MVP", Phases: []string{"planning", "execution", "validation", "done"}, Phase: "planning"}, `{"answer":"Готово.","plan":{"phase":"validation"}}`)
	if plan.Phase != "planning" || plan.PhaseIndex != 0 {
		t.Fatalf("planner skipped a phase: %#v", plan)
	}
	_, plan = applyPlannerCompletion(plan, `{"answer":"Готово.","plan":{"phase":"execution"}}`)
	if plan.Phase != "planning" || plan.PhaseIndex != 0 {
		t.Fatalf("planner changed a phase: %#v", plan)
	}
	_, plan = applyPlannerCompletion(plan, `{"answer":"Возвращаю.","plan":{"phase":"planning"}}`, "Вернись в planning, нужно доработать требования.")
	if plan.Phase != "planning" || plan.PhaseIndex != 0 {
		t.Fatalf("planner changed a phase on return request: %#v", plan)
	}
}

func TestProjectDashboardRejectsDirectPhaseSwitch(t *testing.T) {
	agent := New(&fakeCompleter{})
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "Текстовое ТЗ", Phases: []string{"planning", "execution", "validation", "done"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "switch_task_phase", Phase: "execution"}); err == nil || !strings.Contains(err.Error(), "запрещено") {
		t.Fatalf("direct switch must be rejected, err=%v", err)
	}
}

func TestInvariantsAreSeparateFromDialogueAndIncludedAsMandatoryRules(t *testing.T) {
	client := &fakeCompleter{answer: "Отказываюсь от MySQL: это нарушает инвариант. Могу предложить PostgreSQL."}
	agent := New(client)
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "Сделать API", Phases: []string{"planning", "done"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_invariant", Invariant: models.Invariant{Scope: "task", Rule: "Использовать только PostgreSQL."}}); err != nil {
		t.Fatal(err)
	}
	result, err := agent.Respond(context.Background(), "session", "Давай вместо этого используем MySQL")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Answer, "нарушает инвариант") || len(result.Task.TaskInvariants) != 1 {
		t.Fatalf("conflict result = %#v", result)
	}
	if len(client.requests) != 1 || !strings.Contains(client.requests[0][1].Content, "ОБЯЗАТЕЛЬНЫЕ ИНВАРИАНТЫ") || !strings.Contains(client.requests[0][1].Content, "PostgreSQL") || !strings.Contains(client.requests[0][1].Content, "откажись") {
		t.Fatalf("invariants were not supplied as mandatory context: %#v", client.requests)
	}
	for _, message := range result.Messages {
		if strings.Contains(message.Content, "ОБЯЗАТЕЛЬНЫЕ ИНВАРИАНТЫ") {
			t.Fatalf("invariant leaked into dialogue: %#v", result.Messages)
		}
	}
}

func TestStateInvariantBlocksAutomaticTransitionUntilApproval(t *testing.T) {
	_, task := applyPlannerCompletion(models.TaskState{
		Goal: "MVP", Phases: []string{"planning", "execution"}, Phase: "planning", ExpectedAction: "Подтвердить план",
		RequireApprovalForTransition: true,
	}, `{"answer":"Перехожу.","plan":{"phase":"execution"}}`, "Продолжай без подтверждения")
	if task.Phase != "planning" {
		t.Fatalf("transition without approval = %#v", task)
	}
	_, task = applyPlannerCompletion(task, `{"answer":"Перехожу.","plan":{"phase":"execution"}}`, "Подтверждаю план")
	if task.Phase != "planning" {
		t.Fatalf("model transition with approval = %#v", task)
	}
}

func TestPlannerCanBeDisabledWithoutDiscardingTaskState(t *testing.T) {
	client := &fakeCompleter{answer: "Обычный ответ, не JSON"}
	agent := New(client)
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "MVP", Phases: []string{"planning", "done"}, CurrentStep: "Собрать требования"}}); err != nil {
		t.Fatal(err)
	}
	disabled, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_planner_mode", PlannerMode: "disabled"})
	if err != nil || disabled.PlannerMode != "disabled" || disabled.Task.CurrentStep != "Собрать требования" {
		t.Fatalf("disabled task = %#v, err=%v", disabled.Task, err)
	}
	result, err := agent.Respond(context.Background(), "session", "Расскажи про Android-разработку")
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "Обычный ответ, не JSON" || result.PlannerMode != "disabled" {
		t.Fatalf("response = %#v", result)
	}
	for _, message := range client.requests[0] {
		if strings.Contains(message.Content, "агент-проектировщик") {
			t.Fatalf("planner prompt must be absent: %#v", client.requests[0])
		}
	}
}

func TestPlannerModeCanBeChangedBeforeAnyTaskExists(t *testing.T) {
	agent := New(&fakeCompleter{})
	state, err := agent.ApplyContextCommand("empty-session", models.ContextCommand{Action: "set_planner_mode", PlannerMode: "disabled"})
	if err != nil || state.PlannerMode != "disabled" || state.Task.Goal != "" {
		t.Fatalf("empty-chat planner mode = %#v, err=%v", state, err)
	}
}

func TestGlobalInvariantSurvivesNewSession(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "invariants.json"))
	first, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommandForUser("user", "one", models.ContextCommand{Action: "save_invariant", Invariant: models.Invariant{Scope: "global", Rule: "Не транслитерировать английские термины."}}); err != nil {
		t.Fatal(err)
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	state := second.StateForUser("user", "two")
	if len(state.GlobalInvariants) != 1 || !strings.Contains(state.GlobalInvariants[0].Rule, "транслитерировать") {
		t.Fatalf("global invariants = %#v", state.GlobalInvariants)
	}
}

func TestPauseInFlightInterruptsPlannerBeforeProviderWork(t *testing.T) {
	client := &fakeCompleter{answer: "Не должен быть получен"}
	agent := New(client)
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "ТЗ", Phases: []string{"planning", "done"}}}); err != nil {
		t.Fatal(err)
	}
	result := make(chan models.AgentResponse, 1)
	failure := make(chan error, 1)
	go func() {
		response, err := agent.Respond(context.Background(), "session", "Подготовь черновик ТЗ")
		if err != nil {
			failure <- err
			return
		}
		result <- response
	}()

	var interrupted bool
	deadline := time.Now().Add(300 * time.Millisecond)
	for time.Now().Before(deadline) {
		if response, ok := agent.PauseInFlight("session", "session"); ok {
			interrupted = response.Task.Status == models.TaskPaused
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !interrupted {
		t.Fatal("pause did not interrupt active planner work")
	}
	select {
	case err := <-failure:
		t.Fatal(err)
	case response := <-result:
		if response.Task.Status != models.TaskPaused || response.PendingMessage != "Подготовь черновик ТЗ" || !strings.Contains(response.Answer, "приостановлена") || len(client.requests) != 0 {
			t.Fatalf("pause response = %#v, calls=%d", response, len(client.requests))
		}
	case <-time.After(time.Second):
		t.Fatal("planner did not finish after pause")
	}
}

func TestTaskStateMachineOnlyMovesToAdjacentPhasesAndPersists(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "agent-history.json"))
	first, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "Проверить задачу", Phases: []string{"planning", "execution", "validation", "done"}}}); err != nil {
		t.Fatal(err)
	}
	advanced, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "approve_plan"})
	if err != nil || advanced.Task.Phase != "execution" || advanced.Task.PhaseIndex != 1 || advanced.Task.Status != models.TaskActive {
		t.Fatalf("advanced state = %#v, err=%v", advanced.Task, err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "complete_implementation"}); err != nil {
		t.Fatal(err)
	}
	completed, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "pass_validation"})
	if err != nil || completed.Task.Phase != "done" || completed.Task.Status != models.TaskDone {
		t.Fatalf("completed state = %#v, err=%v", completed.Task, err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "advance_task"}); err == nil {
		t.Fatal("advance after final phase unexpectedly succeeded")
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if restored := second.State("session").Task; restored.Phase != "done" || restored.Status != models.TaskDone || !restored.PlanApproved || !restored.ImplementationCompleted || !restored.ValidationPassed || !reflect.DeepEqual(restored.Phases, []string{"planning", "execution", "validation", "done"}) {
		t.Fatalf("restored task = %#v", restored)
	}
}

func TestTaskLifecycleRejectsInvalidTransitionsAndKeepsState(t *testing.T) {
	agent := New(&fakeCompleter{})
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{
		Goal: "Проверить жизненный цикл", Phases: []string{"planning", "execution", "validation", "done"},
		// A caller must not be able to forge evidence in the configuration.
		PlanApproved: true, ValidationPassed: true,
	}}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "advance_task"}); err == nil || !strings.Contains(err.Error(), "утвердите план") {
		t.Fatalf("planning -> execution without approval must fail, err=%v", err)
	}
	if state := agent.State("session").Task; state.Phase != "planning" || state.PlanApproved || state.ValidationPassed {
		t.Fatalf("failed transition changed or accepted forged state: %#v", state)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "complete_implementation"}); err == nil || !strings.Contains(err.Error(), "Нельзя") {
		t.Fatalf("completion outside execution must fail, err=%v", err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "approve_plan"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "advance_task"}); err == nil || !strings.Contains(err.Error(), "реализацию") {
		t.Fatalf("execution -> validation without completion must fail, err=%v", err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "complete_implementation"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "advance_task"}); err == nil || !strings.Contains(err.Error(), "валидацию") {
		t.Fatalf("validation -> done without validation must fail, err=%v", err)
	}
}

func TestReturningForReworkInvalidatesLaterLifecycleEvidence(t *testing.T) {
	agent := New(&fakeCompleter{})
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "ТЗ", Phases: []string{"planning", "execution", "validation", "done"}}}); err != nil {
		t.Fatal(err)
	}
	for _, action := range []string{"approve_plan", "complete_implementation", "pass_validation"} {
		if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: action}); err != nil {
			t.Fatalf("%s: %v", action, err)
		}
	}
	returned, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "previous_task"})
	if err != nil || returned.Task.Phase != "validation" || returned.Task.Status != models.TaskActive || returned.Task.ValidationPassed {
		t.Fatalf("return from done must reopen validation: %#v, err=%v", returned.Task, err)
	}
	returned, err = agent.ApplyContextCommand("session", models.ContextCommand{Action: "previous_task"})
	if err != nil || returned.Task.Phase != "execution" || returned.Task.ImplementationCompleted || returned.Task.ValidationPassed {
		t.Fatalf("return to execution must invalidate later proof: %#v, err=%v", returned.Task, err)
	}
}

func TestClearKeepsLongTermMemoryButRemovesDialogueAndWorkingMemory(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "agent-history.json"))
	agent, err := NewPersistent(&fakeCompleter{answer: "Ответ"}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "session", "Текущая задача", 2); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryWorking, Value: "Текущая задача"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryLongTerm, Category: "knowledge", Value: "Пишет по-русски"}); err != nil {
		t.Fatal(err)
	}
	if err := agent.Clear("session"); err != nil {
		t.Fatal(err)
	}
	state := agent.State("session")
	if len(state.Messages) != 0 || len(state.Memory.ShortTerm) != 0 || len(state.Memory.Working) != 0 {
		t.Fatalf("cleared task state = %#v", state)
	}
	if got := state.Memory.LongTerm; len(got) != 1 || got[0].Value != "Пишет по-русски" {
		t.Fatalf("long-term memory after clear = %#v", got)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "clear_memory_layer", Layer: models.MemoryLongTerm}); err != nil {
		t.Fatal(err)
	}
	if got := agent.State("session").Memory.LongTerm; len(got) != 0 {
		t.Fatalf("long-term memory after separate clear = %#v", got)
	}
}

func (f *fakeCompleter) CompleteMessages(_ context.Context, messages []models.ChatMessage, _ models.GenerationSettings) (models.ModelCompletion, error) {
	f.requests = append(f.requests, append([]models.ChatMessage(nil), messages...))
	if f.err != nil {
		return models.ModelCompletion{}, f.err
	}
	return models.ModelCompletion{Answer: f.answer, Usage: f.usage}, nil
}

func TestRespondReportsExactProviderUsageAndCumulativeCost(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ", usage: models.ModelUsage{InputTokens: 120, OutputTokens: 30, CacheMissTokens: 120}}
	agent := New(client)
	result, err := agent.Respond(context.Background(), "session", "Короткий вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if result.Tokens.RequestTokens != 120 || result.Tokens.ResponseTokens != 30 || result.Tokens.CumulativeInputTokens != 120 || result.Tokens.CumulativeOutputTokens != 30 {
		t.Fatalf("tokens = %#v", result.Tokens)
	}
	if result.Tokens.EstimatedRequestTokens == 0 || result.Tokens.CurrentMessageTokens == 0 || result.Tokens.CumulativeCostUSD <= 0 {
		t.Fatalf("incomplete token report = %#v", result.Tokens)
	}
	if result.Tokens.CacheMissTokens != 120 || len(result.RequestMessages) != 2 || result.RequestMessages[1].Content != "Короткий вопрос" {
		t.Fatalf("request details = %#v, tokens = %#v", result.RequestMessages, result.Tokens)
	}
}

func TestRespondRejectsDialogueBeyondContextBeforeProviderCall(t *testing.T) {
	client := &fakeCompleter{answer: "Не должен вызываться"}
	agent := New(client)
	conversation := agent.conversation("session")
	conversation.strategy = models.StrategyBranching
	conversation.ensureRootBranchLocked()
	conversation.branches["root"].messages = make([]models.ChatMessage, 100)
	for i := range conversation.branches["root"].messages {
		conversation.branches["root"].messages[i] = models.ChatMessage{Role: "user", Content: strings.Repeat("я", maxMessageCharacters)}
	}
	if _, err := agent.Respond(context.Background(), "session", "Ещё один вопрос"); !errors.Is(err, ErrContextLimit) {
		t.Fatalf("Respond error = %v, want ErrContextLimit", err)
	}
	if len(client.requests) != 0 {
		t.Fatalf("provider requests = %d, want 0", len(client.requests))
	}
	if got := len(agent.History("session")); got != 100 {
		t.Fatalf("history length = %d, want 100; rejected message must not persist", got)
	}
}

func TestTokenReportDoesNotCountSystemInstructionAsDialogueHistory(t *testing.T) {
	agent := New(&fakeCompleter{})
	if err := agent.Clear("session"); err != nil {
		t.Fatal(err)
	}
	report := agent.TokenReport("session")
	if report.HistoryTokens != 0 || report.CurrentMessageTokens != 0 {
		t.Fatalf("cleared dialogue tokens = %#v, want zero", report)
	}
	if report.EstimatedRequestTokens == 0 {
		t.Fatal("system instruction must remain in full request estimate")
	}
}

func TestRespondBuildsHistoryPerSession(t *testing.T) {
	client := &fakeCompleter{answer: "Первый ответ"}
	agent := New(client)
	first, err := agent.Respond(context.Background(), "session-a", "Первый вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := first.Messages, []models.ChatMessage{{Role: "user", Content: "Первый вопрос"}, {Role: "assistant", Content: "Первый ответ"}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("first history = %#v, want %#v", got, want)
	}

	client.answer = "Второй ответ"
	_, err = agent.Respond(context.Background(), "session-a", "Второй вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := client.requests[1][1:], []models.ChatMessage{
		{Role: "user", Content: "Первый вопрос"}, {Role: "assistant", Content: "Первый ответ"}, {Role: "user", Content: "Второй вопрос"},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("second LLM request = %#v, want %#v", got, want)
	}

	client.answer = "Другой диалог"
	other, err := agent.Respond(context.Background(), "session-b", "Другой вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := other.Messages, []models.ChatMessage{{Role: "user", Content: "Другой вопрос"}, {Role: "assistant", Content: "Другой диалог"}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("second session history = %#v, want %#v", got, want)
	}
}

func TestRespondDoesNotKeepFailedMessage(t *testing.T) {
	client := &fakeCompleter{err: errors.New("provider unavailable")}
	agent := New(client)
	if _, err := agent.Respond(context.Background(), "session", "Вопрос"); err == nil {
		t.Fatal("Respond succeeded, want provider error")
	}
	if got := agent.History("session"); len(got) != 0 {
		t.Fatalf("history after failure = %#v, want empty", got)
	}
}

func TestSlidingWindowKeepsOnlyRecentNWithoutSummaryCall(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.Respond(context.Background(), "session", "Первый факт", 2); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "session", "Второй факт", 2); err != nil {
		t.Fatal(err)
	}
	result, err := agent.Respond(context.Background(), "session", "Третий вопрос", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 2 || result.Messages[0].Content != "Третий вопрос" {
		t.Fatalf("raw history = %#v, want only last two messages", result.Messages)
	}
	if len(client.requests) != 3 {
		t.Fatalf("calls = %d, want one provider request per user message", len(client.requests))
	}
	request := client.requests[2]
	if len(request) != 3 || request[1].Role != "assistant" || request[1].Content != "Ответ" || request[2].Content != "Третий вопрос" {
		t.Fatalf("window request = %#v", request)
	}
}

func TestFactsSurviveWindowAndAreSentAsSeparateSystemBlock(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Цель: сделать приложение привычек", 2, models.StrategyFacts); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Ограничение: только iOS", 2, models.StrategyFacts); err != nil {
		t.Fatal(err)
	}
	result, err := agent.RespondWithStrategy(context.Background(), "session", "Что мы делаем?", 2, models.StrategyFacts)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 2 || len(result.Facts) == 0 {
		t.Fatalf("state = %#v", result)
	}
	if !strings.Contains(client.requests[2][1].Content, "Постоянные факты") || !strings.Contains(client.requests[2][1].Content, "сделать приложение") {
		t.Fatalf("facts prompt = %#v", client.requests[2])
	}
}

func TestMemoryLayersAreSeparateAndOnlyExplicitCommandsPersistWorkingAndLongTerm(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.Respond(context.Background(), "session", "Обсуждаем экран заказа", 2); err != nil {
		t.Fatal(err)
	}
	if state := agent.State("session"); len(state.Memory.Working) != 0 || len(state.Memory.LongTerm) != 0 {
		t.Fatalf("ordinary dialogue must not auto-save explicit layers: %#v", state.Memory)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryWorking, Key: "task", Value: "Экран отслеживания заказа"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "knowledge", Key: "language", Value: "Русский"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "decisions", Key: "map", Value: "Не показывать координаты курьера"}); err != nil {
		t.Fatal(err)
	}

	state := agent.State("session")
	if got := len(state.Memory.ShortTerm); got != 2 {
		t.Fatalf("short-term entries = %d, want dialogue only", got)
	}
	if got := state.Memory.Working; len(got) != 1 || got[0].Key != "task" {
		t.Fatalf("working memory = %#v", got)
	}
	if got := state.Memory.LongTerm; len(got) != 2 || got[0].Category != "decisions" || got[1].Category != "knowledge" {
		t.Fatalf("long-term memory = %#v", got)
	}

	if _, err := agent.Respond(context.Background(), "session", "Что учесть в ответе?", 2); err != nil {
		t.Fatal(err)
	}
	request := client.requests[len(client.requests)-1]
	if !strings.Contains(request[1].Content, "Долговременная память") || !strings.Contains(request[1].Content, "knowledge.language: Русский") {
		t.Fatalf("long-term block = %#v", request)
	}
	if !strings.Contains(request[2].Content, "Рабочая память") || !strings.Contains(request[2].Content, "task: Экран") {
		t.Fatalf("working block = %#v", request)
	}
	if len(request) != 5 || request[3].Role != "assistant" || request[4].Content != "Что учесть в ответе?" {
		t.Fatalf("short-term window in request = %#v", request)
	}
}

func TestMemoryLayersPersistAndShortTermCannotBeWrittenDirectly(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "memory.json"))
	first, err := NewPersistent(&fakeCompleter{answer: "Ответ"}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryWorking, Key: "deadline", Value: "пятница"}); err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "knowledge", Key: "api", Value: "Используем существующее API"}); err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryShortTerm, Key: "x", Value: "y"}); err == nil {
		t.Fatal("short-term memory write succeeded")
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	state := second.State("session")
	if len(state.Memory.ShortTerm) != 0 || len(state.Memory.Working) != 1 || len(state.Memory.LongTerm) != 1 {
		t.Fatalf("restored layers = %#v", state.Memory)
	}
}

func TestProfileIsSentForEveryRequestAndKeepsProfilesIndependent(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	anna := models.UserProfile{Name: "Анна", Style: "деловой", Format: "ровно 3 пункта", Constraints: "без англицизмов"}
	maxim := models.UserProfile{Name: "Максим", Style: "неформальный", Format: "подробный разбор", Constraints: "добавь пример к каждому совету"}
	if _, err := agent.ApplyContextCommand("anna", models.ContextCommand{Action: "set_profile", Profile: anna}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("maxim", models.ContextCommand{Action: "set_profile", Profile: maxim}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "anna", "Как начать изучать Go?"); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "maxim", "Как начать изучать Go?"); err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(client.requests))
	}
	annaPrompt, maximPrompt := client.requests[0][1].Content, client.requests[1][1].Content
	if !strings.Contains(annaPrompt, "Анна") || !strings.Contains(annaPrompt, "ровно 3 пункта") || !strings.Contains(annaPrompt, "без англицизмов") {
		t.Fatalf("Anna profile was not sent: %q", annaPrompt)
	}
	if !strings.Contains(maximPrompt, "Максим") || !strings.Contains(maximPrompt, "подробный разбор") || !strings.Contains(maximPrompt, "добавь пример") {
		t.Fatalf("Maxim profile was not sent: %q", maximPrompt)
	}
	if annaPrompt == maximPrompt {
		t.Fatal("different profiles produced identical model context")
	}
	if got := agent.State("anna").Profile; got.Name != anna.Name || got.Style != anna.Style || got.Format != anna.Format || got.Constraints != anna.Constraints || got.ID == "" {
		t.Fatalf("Anna profile = %#v, want saved values %#v", got, anna)
	}
	if err := agent.Clear("anna"); err != nil {
		t.Fatal(err)
	}
	if got := agent.State("anna").Profile; got.Name != anna.Name || got.ID == "" {
		t.Fatalf("profile after clearing task = %#v, want saved values %#v", got, anna)
	}
}

func TestProfilePersistsAfterRestart(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "profile.json"))
	first, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	want := models.UserProfile{Name: "Ирина", Style: "лаконичный", Format: "список", Constraints: "не более 4 пунктов"}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "set_profile", Profile: want}); err != nil {
		t.Fatal(err)
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if got := second.State("session").Profile; got.Name != want.Name || got.Style != want.Style || got.ID == "" {
		t.Fatalf("restored profile = %#v, want saved values %#v", got, want)
	}
}

func TestNamedProfilesAreSelectedPerSessionAndLongTermMemoryIsGlobalPerUser(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	const userID = "user-a"
	chemistry := models.UserProfile{Name: "Химик", Perspective: "химик-популяризатор: объясняй причины и механизмы", Style: "научно-популярный", Format: "короткий пост"}
	psychology := models.UserProfile{Name: "Психолог", Perspective: "психолог-практик: объясняй поведение без диагнозов", Style: "бережный", Format: "пост с примером"}

	chemistryState, err := agent.ApplyContextCommandForUser(userID, "session-post", models.ContextCommand{Action: "create_profile", Profile: chemistry})
	if err != nil {
		t.Fatal(err)
	}
	chemistryID := chemistryState.ActiveProfileID
	psychologyState, err := agent.ApplyContextCommandForUser(userID, "session-post", models.ContextCommand{Action: "create_profile", Profile: psychology})
	if err != nil {
		t.Fatal(err)
	}
	psychologyID := psychologyState.ActiveProfileID
	if chemistryID == psychologyID || len(psychologyState.Profiles) != 2 {
		t.Fatalf("profiles = %#v", psychologyState.Profiles)
	}
	if _, err := agent.ApplyContextCommandForUser(userID, "session-post", models.ContextCommand{Action: "set_active_profile", ProfileID: chemistryID}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommandForUser(userID, "session-review", models.ContextCommand{Action: "set_active_profile", ProfileID: psychologyID}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.RespondWithUserOptions(context.Background(), userID, "session-post", "Напиши пост о кофеине", 2, models.StrategySlidingWindow, ""); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.RespondWithUserOptions(context.Background(), userID, "session-review", "Напиши пост о кофеине", 2, models.StrategySlidingWindow, ""); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(client.requests[0][1].Content, "химик-популяризатор") || !strings.Contains(client.requests[1][1].Content, "психолог-практик") {
		t.Fatalf("profile prompts = %#v", client.requests)
	}
	if agent.StateForUser(userID, "session-post").ActiveProfileID != chemistryID || agent.StateForUser(userID, "session-review").ActiveProfileID != psychologyID {
		t.Fatal("active profile must belong to a session")
	}
	if _, err := agent.ApplyContextCommandForUser(userID, "session-post", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "knowledge", Key: "audience", Value: "Пишем для начинающих"}); err != nil {
		t.Fatal(err)
	}
	otherSession := agent.StateForUser(userID, "session-review")
	if len(otherSession.Memory.LongTerm) != 1 || otherSession.Memory.LongTerm[0].Value != "Пишем для начинающих" {
		t.Fatalf("global long-term memory = %#v", otherSession.Memory.LongTerm)
	}
	if got := agent.StateForUser("user-b", "other-session").Memory.LongTerm; len(got) != 0 {
		t.Fatalf("another user's memory leaked: %#v", got)
	}
}

func TestUserProfilesAndGlobalMemoryPersistAcrossRestart(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "user-state.json"))
	first, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	const userID = "user-persisted"
	created, err := first.ApplyContextCommandForUser(userID, "first-session", models.ContextCommand{Action: "create_profile", Profile: models.UserProfile{Name: "Экономист", Perspective: "экономист, объясняй стимулы и последствия"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommandForUser(userID, "first-session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "knowledge", Key: "audience", Value: "Предприниматели"}); err != nil {
		t.Fatal(err)
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	state := second.StateForUser(userID, "second-session")
	if len(state.Profiles) != 1 || state.Profiles[0].Name != "Экономист" || len(state.Memory.LongTerm) != 1 || state.Memory.LongTerm[0].Value != "Предприниматели" {
		t.Fatalf("restored user state = %#v", state)
	}
	if _, err := second.ApplyContextCommandForUser(userID, "second-session", models.ContextCommand{Action: "set_active_profile", ProfileID: created.ActiveProfileID}); err != nil {
		t.Fatal(err)
	}
	if got := second.StateForUser(userID, "second-session").Profile.Name; got != "Экономист" {
		t.Fatalf("active profile after restart = %q", got)
	}
}

func TestSelectedDialogueMessageIsStoredWholeWithoutUserKey(t *testing.T) {
	agent := New(&fakeCompleter{answer: "Ответ агента"})
	if _, err := agent.Respond(context.Background(), "session", "Не показывать карту курьера", 2); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryWorking, Value: "Не показывать карту курьера"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryLongTerm, Category: "decisions", Value: "Ответ агента"}); err != nil {
		t.Fatal(err)
	}
	state := agent.State("session")
	if got := state.Memory.Working; len(got) != 1 || got[0].Key != "message-1" || got[0].Value != "Не показывать карту курьера" {
		t.Fatalf("working memory = %#v", got)
	}
	if got := state.Memory.LongTerm; len(got) != 1 || got[0].Key != "message-2" || got[0].Category != "decisions" || got[0].Value != "Ответ агента" {
		t.Fatalf("long-term memory = %#v", got)
	}
}

func TestModelSelectionPersistsAndUsesSelectedModel(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ Pro"}
	agent := New(client)
	result, err := agent.RespondWithOptions(context.Background(), "session", "Составь план", 2, models.StrategySlidingWindow, models.DeepSeekProModel)
	if err != nil {
		t.Fatal(err)
	}
	if result.Model != models.DeepSeekProModel || len(client.models) != 1 || client.models[0] != models.DeepSeekProModel {
		t.Fatalf("selected model state=%q calls=%#v", result.Model, client.models)
	}
	if state := agent.State("session"); state.Model != models.DeepSeekProModel {
		t.Fatalf("model after response = %q", state.Model)
	}
}

func TestBranchingCreatesIndependentChildrenFromCheckpoint(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Общая задача", 2, models.StrategyBranching); err != nil {
		t.Fatal(err)
	}
	state, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "checkpoint", Name: "до решения"})
	if err != nil || len(state.Checkpoints) != 1 {
		t.Fatalf("checkpoint state = %#v, err=%v", state, err)
	}
	checkpointID := state.Checkpoints[0].ID
	state, err = agent.ApplyContextCommand("session", models.ContextCommand{Action: "create_branch", CheckpointID: checkpointID, Name: "вариант A"})
	if err != nil {
		t.Fatal(err)
	}
	branchA := state.ActiveBranchID
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Продолжение A", 0, models.StrategyBranching); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "create_branch", CheckpointID: checkpointID, Name: "вариант B"}); err != nil {
		t.Fatal(err)
	}
	if got := agent.History("session"); len(got) != 2 || got[0].Content != "Общая задача" {
		t.Fatalf("new branch history = %#v", got)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "switch_branch", BranchID: branchA}); err != nil {
		t.Fatal(err)
	}
	if got := agent.History("session"); len(got) != 4 || got[2].Content != "Продолжение A" {
		t.Fatalf("restored branch history = %#v", got)
	}
}
