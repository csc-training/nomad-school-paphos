! Main solver routines for heat equation solver
module core
  use heat

contains

  ! Exchange the boundary data between MPI tasks
  subroutine exchange(field0, parallel)
    use mpi

    implicit none

    type(field), target, intent(inout) :: field0
    type(parallel_data), intent(inout) :: parallel

    real(dp), pointer, contiguous, dimension(:,:,:) :: data

    integer :: buf_size

    integer :: requests(12)
    integer :: ierr

    data => field0%data


    ! Post all receives:
    ! z-direction
    buf_size = (field0%nx + 2) * (field0%ny + 2)
    call mpi_irecv(data(:, :, 0), buf_size, MPI_DOUBLE_PRECISION,&
         & parallel%ngbrs(1, 1), 11, parallel%comm, requests(1), ierr)
    call mpi_irecv(data(:, :, field0%nz + 1), buf_size, MPI_DOUBLE_PRECISION,&
         & parallel%ngbrs(1, 2), 11, parallel%comm, requests(2), ierr)

    ! y-direction
    buf_size = (field0%nx + 2) * (field0%nz + 2)
    call mpi_irecv(parallel%recvbuffer(2, 1)%data, buf_size, MPI_DOUBLE_PRECISION,&
         & parallel%ngbrs(2, 1), 11, parallel%comm, requests(5), ierr)
    call mpi_irecv(parallel%recvbuffer(2, 2)%data, buf_size, MPI_DOUBLE_PRECISION,&
         & parallel%ngbrs(2, 2), 11, parallel%comm, requests(6), ierr)

    ! x-direction
    buf_size = (field0%ny + 2) * (field0%nz + 2)
    call mpi_irecv(parallel%recvbuffer(3, 1)%data, buf_size, MPI_DOUBLE_PRECISION,&
         & parallel%ngbrs(3, 1), 11, parallel%comm, requests(9), ierr)
    call mpi_irecv(parallel%recvbuffer(3, 2)%data, buf_size, MPI_DOUBLE_PRECISION,&
         & parallel%ngbrs(3, 2), 11, parallel%comm, requests(10), ierr)

    ! Send
    ! z-direction
    buf_size = (field0%nx + 2) * (field0%ny + 2)
    call mpi_isend(data(:, :, 1), buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%ngbrs(1, 1), 11, parallel%comm, requests(3), ierr)
    call mpi_isend(data(:, :, field0%nz), buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%ngbrs(1, 2), 11, parallel%comm, requests(4), ierr)

    ! y-direction
    buf_size = (field0%nx + 2) * (field0%nz + 2)
    if (parallel%ngbrs(2, 1) /= MPI_PROC_NULL) then
       ! copy send buffer
       parallel%sendbuffer(2, 1)%data = data(:, 1, :)
    end if
    call mpi_isend(parallel%sendbuffer(2, 1)%data, buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%ngbrs(2, 1), 11, parallel%comm, requests(7), ierr)

    if (parallel%ngbrs(2, 2) /= MPI_PROC_NULL) then
       ! copy send buffer
       parallel%sendbuffer(2, 2)%data = data(:, field0%ny, :)
    end if
    call mpi_isend(parallel%sendbuffer(2, 2)%data, buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%ngbrs(2, 2), 11, parallel%comm, requests(8), ierr)

    ! x-direction
    buf_size = (field0%ny + 2) * (field0%nz + 2)
    if (parallel%ngbrs(3, 1) /= MPI_PROC_NULL) then
       ! copy send buffer
       parallel%sendbuffer(3, 1)%data = data(1, :, :)
    end if
    call mpi_isend(parallel%sendbuffer(3, 1)%data, buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%ngbrs(3, 1), 11, parallel%comm, requests(11), ierr)

    if (parallel%ngbrs(3, 2) /= MPI_PROC_NULL) then
       ! copy send buffer
       parallel%sendbuffer(3, 2)%data = data(field0%nx, :, :)
    end if
    call mpi_isend(parallel%sendbuffer(3, 2)%data, buf_size, MPI_DOUBLE_PRECISION, &
         & parallel%ngbrs(3, 2), 11, parallel%comm, requests(12), ierr)

    call mpi_waitall(12, requests, MPI_STATUSES_IGNORE, ierr)

    ! copy recv buffers
    if (parallel%ngbrs(2, 1) /= MPI_PROC_NULL) then
       data(:, 0, :) = parallel%recvbuffer(2, 1)%data 
    end if
    if (parallel%ngbrs(2, 2) /= MPI_PROC_NULL) then
       data(:, field0%ny + 1, :) = parallel%recvbuffer(2, 2)%data 
    end if
    if (parallel%ngbrs(3, 1) /= MPI_PROC_NULL) then
       data(0, :, :) = parallel%recvbuffer(3, 1)%data 
    end if
    if (parallel%ngbrs(3, 2) /= MPI_PROC_NULL) then
       data(field0%nx + 1, :, :) = parallel%recvbuffer(3, 2)%data 
    end if

  end subroutine exchange

  ! Compute one time step of temperature evolution
  ! Arguments:
  !   curr (type(field)): current temperature values
  !   prev (type(field)): values from previous time step
  !   a (real(dp)): update equation constant
  !   dt (real(dp)): time step value
  subroutine evolve(curr, prev, a, dt)

    implicit none

    type(field), target, intent(inout) :: curr, prev
    real(dp) :: a, dt
    integer :: i, j, k, nx, ny, nz
    real(dp) :: inv_dx2, inv_dy2, inv_dz2
    ! variables for memory access outside of a type
    real(dp), pointer, contiguous, dimension(:,:,:) :: currdata, prevdata

    nx = curr%nx
    ny = curr%ny
    nz = curr%nz
    inv_dx2 = 1.0 / curr%dx**2
    inv_dy2 = 1.0 / curr%dy**2
    inv_dz2 = 1.0 / curr%dz**2
    currdata => curr%data
    prevdata => prev%data

    !$omp parallel do private(i,j,k) collapse(2) schedule(static)
    do k = 1, nz
       do j = 1, ny
          !$omp simd
          do i = 1, nx
             currdata(i, j, k) = prevdata(i, j, k) + a * dt * &
                  & ((prevdata(i-1, j, k) - 2.0 * prevdata(i, j, k) + &
                  &   prevdata(i+1, j, k)) * inv_dx2 + &
                  &  (prevdata(i, j-1, k) - 2.0 * prevdata(i, j, k) + &
                  &   prevdata(i, j+1, k)) * inv_dy2 + &
                  &  (prevdata(i, j, k-1) - 2.0 * prevdata(i, j, k) + &
                  &   prevdata(i, j, k+1)) * inv_dz2)

          end do
       end do
    end do
    !$omp end parallel do

  end subroutine evolve

end module core
