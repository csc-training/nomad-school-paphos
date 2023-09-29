! Heat equation solver in 2D.

program heat_solve
  use heat
  use core
  use setup
  use utilities
  use mpi

  implicit none

  real(dp), parameter :: a = 0.5 ! Diffusion constant
  type(field) :: current, previous    ! Current and previus temperature fields

  real(dp) :: dt     ! Time step
  integer :: nsteps       ! Number of time steps

  type(parallel_data) :: parallelization
  integer :: ierr, provided

  integer :: iter

  real(dp) :: average_temp   !  Average temperature

  real(kind=dp) :: start_time, stop_time ! Timers
  real(kind=dp) :: t_mpi = 0.0, t_comp = 0.0
  real(kind=dp) :: start_mpi, start_comp, gflops

  call mpi_init_thread(MPI_THREAD_SERIALIZED, provided, ierr)

  call initialize(current, previous, nsteps, parallelization)

  average_temp = average(current)
  if (parallelization % rank == 0) then
     write(*,'(A,F9.6)') 'Average temperature at start: ', average_temp
  end if

  ! Largest stable time step
  dt = current%dx**2 * current%dy**2 * current%dz**2 / &
       & (2.0 * a * (current%dx**2 + current%dy**2 + current%dz**2))

  ! Main iteration loop, save a picture every
  ! image_interval steps

  start_time =  mpi_wtime()

  do iter = 1, nsteps
     start_mpi = mpi_wtime()
     call exchange(previous, parallelization)
     t_mpi = t_mpi + mpi_wtime() - start_mpi
     start_comp = mpi_wtime()
     call evolve(current, previous, a, dt)
     t_comp = t_comp + mpi_wtime() - start_comp
     call swap_fields(current, previous)
  end do

  stop_time = mpi_wtime()

  ! Average temperature for reference
  average_temp = average(previous)

  if (parallelization % rank == 0) then
     gflops = current%nx_full*nsteps*14.0e-9
     gflops = gflops * current%ny_full
     gflops = gflops * current%nz_full / (stop_time - start_time)
     write(*,'(A,F7.3,A,F7.3)') 'Iteration took ', stop_time - start_time, & 
                                ' seconds. GFLOP/s: ', gflops
     write(*,'(A,F7.3,A)') '   MPI         ', t_mpi, ' s.'
     write(*,'(A,F7.3,A)') '   Compute     ', t_comp, ' s.'
     write(*,'(A,F9.6)') 'Average temperature: ',  average_temp
     if (command_argument_count() == 0) then
         write(*,'(A,F9.6)') 'Reference value with default arguments: ', 63.834223_dp
     end if
     gflops = current%nx_full*nsteps*14.0e-9
     gflops = gflops * current%ny_full
     gflops = gflops * current%nz_full
  end if

  call finalize(current, previous)

  call mpi_finalize(ierr)

end program heat_solve
