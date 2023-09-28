! Field metadata for heat equation solver
module heat
  use iso_fortran_env, only : REAL64
  use mpi

  implicit none

  integer, parameter :: dp = REAL64
  real(dp), parameter :: DX = 0.01, DY = 0.01, DZ = 0.01  ! Fixed grid spacing

  type :: field
     integer :: nx          ! local dimension of the field
     integer :: ny
     integer :: nz
     integer :: nx_full     ! global dimension of the field
     integer :: ny_full
     integer :: nz_full
     real(dp) :: dx
     real(dp) :: dy
     real(dp) :: dz
     real(dp), dimension(:,:,:), allocatable :: data
  end type field

  type :: communication_buffer
     real(dp), dimension(:,:), allocatable :: data
  end type communication_buffer

  type :: parallel_data
     integer :: size
     integer :: rank
     integer :: dims(3) = (/0, 0, 0/)
     integer :: coords(3) = (/0, 0, 0/)
     integer :: ngbrs(3, 2) ! Ranks of neighbouring MPI tasks
     integer :: num_threads
     integer :: comm  ! Cartesian communicator

     ! Communication buffers
     type(communication_buffer), dimension(3, 2) :: sendbuffer 
     type(communication_buffer), dimension(3, 2) :: recvbuffer 
  end type parallel_data

contains
  ! Initialize the field type metadata
  ! Arguments:
  !   field0 (type(field)): input field
  !   nx, ny, dx, dy: field dimensions and spatial step size
  subroutine set_field_dimensions(field0, nx, ny, nz, parallel)
    implicit none

    type(field), intent(out) :: field0
    integer, intent(in) :: nx, ny, nz
    type(parallel_data), intent(in) :: parallel

    integer :: nx_local, ny_local, nz_local

    nx_local = nx / parallel%dims(3)
    ny_local = ny / parallel%dims(2)
    nz_local = nz / parallel%dims(1)

    field0%dx = DX
    field0%dy = DY
    field0%dz = DZ
    field0%nx = nx_local
    field0%ny = ny_local
    field0%nz = nz_local
    field0%nx_full = nx
    field0%ny_full = ny
    field0%nz_full = nz

  end subroutine set_field_dimensions

  subroutine parallel_setup(parallel, nx, ny, nz)
#ifdef _OPENMP
    use omp_lib
#endif

    implicit none

    type(parallel_data), intent(out) :: parallel
    integer, intent(in), optional :: nx, ny, nz

    integer, parameter :: ndims = 3
    logical :: periods(3) = (/.false., .false., .false./)
    logical, parameter :: reorder = .true.
    integer :: i
    integer :: nx_local, ny_local, nz_local
    integer :: ierr

    call mpi_comm_size(MPI_COMM_WORLD, parallel%size, ierr)

    call mpi_dims_create(parallel%size, ndims, parallel%dims, ierr)
    call mpi_cart_create(MPI_COMM_WORLD, ndims, parallel%dims, periods, reorder, & 
                         parallel%comm, ierr)
    call mpi_comm_rank(parallel%comm, parallel%rank, ierr);
    call mpi_cart_get(parallel%comm, ndims, parallel%dims, periods, parallel%coords, ierr)

    if (present(nz)) then
       nz_local = nz / parallel%dims(1)
       if (nz_local * parallel%dims(1) /= nz) then
          write(*,*) 'Cannot divide grid evenly to processors'
          call mpi_abort(MPI_COMM_WORLD, -2, ierr)
       end if
    end if
    if (present(ny)) then
       ny_local = ny / parallel%dims(2)
       if (ny_local * parallel%dims(2) /= ny) then
          write(*,*) 'Cannot divide grid evenly to processors'
          call mpi_abort(MPI_COMM_WORLD, -2, ierr)
       end if
    end if
    if (present(nx)) then
       nx_local = nx / parallel%dims(3)
       if (nx_local * parallel%dims(3) /= nx) then
          write(*,*) 'Cannot divide grid evenly to processors'
          call mpi_abort(MPI_COMM_WORLD, -2, ierr)
       end if
    end if

    do i=1, 3
       call mpi_cart_shift(parallel%comm, i-1, 1, parallel%ngbrs(i, 1), & 
                                                  parallel%ngbrs(i, 2), ierr)
    end do

    ! allocate communication buffers
    allocate(parallel%sendbuffer(3,1)%data(ny_local + 2, nz_local + 2))
    allocate(parallel%sendbuffer(3,2)%data(ny_local + 2, nz_local + 2))
    allocate(parallel%sendbuffer(2,1)%data(nx_local + 2, nz_local + 2))
    allocate(parallel%sendbuffer(2,2)%data(nx_local + 2, nz_local + 2))
    allocate(parallel%sendbuffer(1,1)%data(nx_local + 2, ny_local + 2))
    allocate(parallel%sendbuffer(1,2)%data(nx_local + 2, ny_local + 2))
    allocate(parallel%recvbuffer(3,1)%data(ny_local + 2, nz_local + 2))
    allocate(parallel%recvbuffer(3,2)%data(ny_local + 2, nz_local + 2))
    allocate(parallel%recvbuffer(2,1)%data(nx_local + 2, nz_local + 2))
    allocate(parallel%recvbuffer(2,2)%data(nx_local + 2, nz_local + 2))
    allocate(parallel%recvbuffer(1,1)%data(nx_local + 2, ny_local + 2))
    allocate(parallel%recvbuffer(1,2)%data(nx_local + 2, ny_local + 2))

    parallel%num_threads = 1
#ifdef _OPENMP
    parallel%num_threads = omp_get_max_threads()
#endif


  end subroutine parallel_setup

end module heat
