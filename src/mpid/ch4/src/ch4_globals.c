/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/* All global ADI data structures need to go in this file */
/* reference them with externs from other files           */

#include <mpidimpl.h>
#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "ch4_impl.h"
#include "utarray.h"

MPIDI_CH4_Global_t MPIDI_global;

MPIDI_NM_funcs_t *MPIDI_NM_func;
MPIDI_NM_native_funcs_t *MPIDI_NM_native_func;

MPID_Thread_mutex_t MPIR_THREAD_VCI_HANDLE_POOL_MUTEXES[MPIR_REQUEST_NUM_POOLS];

/* progress */

/* NOTE: MPL_TLS may be empty if it is unavailable. Since we just need ensure global
 * progress happen, so some race condition or even corruption can be tolerated.  */
MPL_TLS int global_vci_poll_count = 0;

MPL_TLS int no_progress_counter = 0;

/* ** HACK **
 * Hack to workaround an Intel compiler bug on macOS. Touching
 * global_vci_poll_count in this file forces the compiler to allocate
 * it as TLS. See https://github.com/pmodels/mpich/issues/3437.
 */
int _dummy_touch_tls(void);
int _dummy_touch_tls(void)
{
    return global_vci_poll_count;
}

/* PVAR */
unsigned PVAR_LEVEL_posted_recvq_length ATTRIBUTE((unused));
unsigned PVAR_LEVEL_unexpected_recvq_length ATTRIBUTE((unused));
unsigned long long PVAR_COUNTER_posted_recvq_match_attempts ATTRIBUTE((unused));
unsigned long long PVAR_COUNTER_unexpected_recvq_match_attempts ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_time_failed_matching_postedq ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_time_matching_unexpectedq ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_winlock_getlocallock ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_wincreate_allgather ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_amhdr_set ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_put ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_put_ack ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_get ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_get_ack ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_cas ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_cas_ack ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_acc ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_get_acc ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_acc_ack ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_get_acc_ack ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_win_ctrl ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_put_dt ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_put_dt_ack ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_put_data ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_acc_dt ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_get_acc_dt ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_acc_dt_ack ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_get_acc_dt_ack ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_acc_data ATTRIBUTE((unused));
MPIR_T_pvar_timer_t PVAR_TIMER_rma_targetcb_get_acc_data ATTRIBUTE((unused));

#ifdef MPID_DEVICE_DEFINES_THREAD_CS
pthread_mutex_t MPIDI_Mutex_lock[MPIDI_NUM_LOCKS];
#endif

#ifdef HAVE_SIGNAL
void MPIDI_sigusr1_handler(int sig)
{
    MPIDI_global.sigusr1_count++;
    if (MPIDI_global.prev_sighandler)
        MPIDI_global.prev_sighandler(sig);
}
#endif

int MPID_Abort(MPIR_Comm * comm, int mpi_errno, int exit_code, const char *error_msg)
{
    MPIR_FUNC_ENTER;

    char world_str[MPI_MAX_ERROR_STRING] = "";
    if (MPIR_Process.comm_world) {
        int rank = MPIR_Process.rank;
        snprintf(world_str, sizeof(world_str), " on node %d", rank);
    }

    char comm_str[MPI_MAX_ERROR_STRING] = "";
    if (comm) {
        int rank = comm->rank;
        int context_id = comm->context_id;
        snprintf(comm_str, sizeof(comm_str), " (rank %d in comm %d)", rank, context_id);
    }

    if (!error_msg)
        error_msg = "Internal error";

    char sys_str[MPI_MAX_ERROR_STRING + 5] = "";
    if (mpi_errno != MPI_SUCCESS) {
        char msg[MPI_MAX_ERROR_STRING] = "";
        MPIR_Err_get_string(mpi_errno, msg, MPI_MAX_ERROR_STRING, NULL);
        snprintf(sys_str, sizeof(sys_str), " (%s)", msg);
    }

    char error_str[3 * MPI_MAX_ERROR_STRING + 128];
    snprintf(error_str, sizeof(error_str), "Abort(%d)%s%s: %s%s\n",
             exit_code, world_str, comm_str, error_msg, sys_str);
    MPL_error_printf("%s", error_str);

#ifdef HAVE_DEBUGGER_SUPPORT
    MPIR_Debugger_set_aborting(error_msg);
#endif
    MPIR_FUNC_EXIT;
    fflush(stderr);
    fflush(stdout);

    if (MPIR_CVAR_COREDUMP_ON_ABORT) {
        abort();
    }

    if (NULL == comm || (MPIR_Comm_size(comm) == 1 && comm->comm_kind == MPIR_COMM_KIND__INTRACOMM))
        MPL_exit(exit_code);

    if (comm != MPIR_Process.comm_world) {
        MPIDIG_am_comm_abort(comm, exit_code);
    } else {
        MPIR_pmi_abort(exit_code, error_msg);
    }
    return 0;
}

static UT_array *mpidi_failed_procs;

static int mpidi_record_failed_proc(MPIR_Lpid failed_lpid)
{
    int is_new = 1;

    if (!mpidi_failed_procs) {
        utarray_new(mpidi_failed_procs, &ut_int_icd, MPL_MEM_OTHER);
    }

    for (int i = 0; i < utarray_len(mpidi_failed_procs); i++) {
        MPIR_Lpid *p = (MPIR_Lpid *) utarray_eltptr(mpidi_failed_procs, i);
        if (*p == failed_lpid) {
            is_new = 0;
            break;
        }
    }

    if (is_new) {
        utarray_push_back(mpidi_failed_procs, &failed_lpid, MPL_MEM_OTHER);
    }

    return is_new;
}

static bool mpidi_comm_contains_failed_lpid(MPIR_Comm * comm, MPIR_Lpid failed_lpid)
{
    MPIR_Group *group = NULL;

    if (comm->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
        group = comm->local_group;
    } else {
        group = comm->remote_group;
    }

    if (!group) {
        return false;
    }

    for (int i = 0; i < group->size; i++) {
        if (MPIR_Group_rank_to_lpid(group, i) == failed_lpid) {
            return true;
        }
    }

    return false;
}

static bool mpidi_parse_failed_proc_token(const char *token, int *rank)
{
    bool success = true;

    char *tmp = MPL_strdup(token);
    if (!tmp) goto fn_fail;

    char *s = tmp;
    while (isspace(*s)) s++;
    if (*s == '\0') goto fn_fail;

    char *end = s + strlen(s) - 1;
    while (end >= s && isspace(*end)) *end-- = '\0';

    char *parse_end = NULL;
    long value = strtol(s, &parse_end, 10);
    if (*parse_end != '\0' || value < 0 || value > INT_MAX) goto fn_fail;

    *rank = value;

  fn_exit:
    if (tmp) MPL_free(tmp);
    return success;

  fn_fail:
    success = false;
    goto fn_exit;
}

static void mpidi_decrement_revoke_waiting_for_failed_lpid(MPIR_Lpid failed_lpid)
{
    MPID_THREAD_CS_ENTER(VCI, MPIDIU_THREAD_UTIL_MUTEX);

    MPIR_Comm *comm = MPIDI_global.waiting_revoke_comm_list;
    while (comm) {
        MPIR_Comm *next = MPIDI_COMM(comm, waiting_revoke_next);

        MPIR_Assert(MPIDI_COMM(comm, waiting_for_revoke) > 0);
        if (mpidi_comm_contains_failed_lpid(comm, failed_lpid)) {
            MPIDI_COMM(comm, waiting_for_revoke)--;
            if (MPIDI_COMM(comm, waiting_for_revoke) == 0) {
                MPIDI_waiting_revoke_remove_impl(comm);
            }
        }

        comm = next;
    }

    MPID_THREAD_CS_EXIT(VCI, MPIDIU_THREAD_UTIL_MUTEX);
}

int MPIDI_check_for_failed_procs(void)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_FUNC_ENTER;

    /* FIXME: Currently this only handles failed processes in
     * comm_world.  We need to fix hydra to include the pgid along
     * with the rank, then we need to create the failed group from
     * something bigger than comm_world. */

    char *failed_procs_string = MPIR_pmi_get_jobattr("PMI_dead_processes");

    if (failed_procs_string && failed_procs_string[0] != '\0') {
        MPL_DBG_MSG_FMT(MPIDI_CH4_DBG_GENERAL, VERBOSE,
                        (MPL_DBG_FDEST,
                         "Received proc fail notification: %s", failed_procs_string));

        const char *delim = ",";
        char *token = strtok(failed_procs_string, delim);
        while (token != NULL) {
            int rank;

            if (!mpidi_parse_failed_proc_token(token, &rank)) {
                MPL_DBG_MSG_FMT(MPIDI_CH4_DBG_GENERAL, TYPICAL,
                                (MPL_DBG_FDEST,
                                 "Ignoring malformed PMI_dead_processes token: %s", token));
                token = strtok(NULL, delim);
                continue;
            }

            MPIR_Lpid failed_lpid = MPIR_comm_rank_to_lpid(MPIR_Process.comm_world, rank);
            if (mpidi_record_failed_proc(failed_lpid)) {
                mpidi_decrement_revoke_waiting_for_failed_lpid(failed_lpid);
            }
            token = strtok(NULL, delim);
        }
    }

    if (failed_procs_string) {
        MPL_free(failed_procs_string);
    }

    MPIR_FUNC_EXIT;
    return mpi_errno;
}

MPL_dbg_class MPIDI_CH4_DBG_GENERAL;

#ifdef MPL_USE_DBG_LOGGING
MPL_dbg_class MPIDI_CH4_DBG_GENERAL;
MPL_dbg_class MPIDI_CH4_DBG_MAP;
MPL_dbg_class MPIDI_CH4_DBG_COMM;
MPL_dbg_class MPIDI_CH4_DBG_MEMORY;
MPL_dbg_class MPIDI_CH4_DBG_IPC;
#endif
