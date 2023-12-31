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
  integer, parameter :: image_interval = 1500 ! Image output interval

  type(parallel_data) :: parallelization
  integer :: ierr

  integer :: iter

  real(dp) :: average_temp   !  Average temperature

  real(kind=dp) :: start_time, stop_time ! Timers

  call mpi_init(ierr)

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

  call enter_data(current, previous)

  do iter = 1, nsteps
     call exchange(previous, parallelization)
     call evolve(current, previous, a, dt)
     call swap_fields(current, previous)
  end do

  call exit_data(current, previous)
  stop_time = mpi_wtime()

  ! Average temperature for reference
  average_temp = average(previous)

  if (parallelization % rank == 0) then
     write(*,'(A,F7.3,A)') 'Iteration took ', stop_time - start_time, ' seconds.'
     write(*,'(A,F9.6)') 'Average temperature: ',  average_temp
     if (command_argument_count() == 0) then
         write(*,'(A,F9.6)') 'Reference value with default arguments: ', 63.834223_dp
     end if
  end if

  call finalize(current, previous)

  call mpi_finalize(ierr)

end program heat_solve
