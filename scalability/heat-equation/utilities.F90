! Utility routines for heat equation solver
!   NOTE: This file does not need to be edited!
module utilities
  use heat

contains

  ! Swap the data fields of two variables of type field
  ! Arguments:
  !   curr, prev (type(field)): the two variables that are swapped
  subroutine swap_fields(curr, prev)

    implicit none

    type(field), intent(inout) :: curr, prev
    real(dp), allocatable, dimension(:,:,:) :: tmp

    call move_alloc(curr%data, tmp)
    call move_alloc(prev%data, curr%data)
    call move_alloc(tmp, prev%data)
  end subroutine swap_fields

  ! Copy the data from one field to another
  ! Arguments:
  !   from_field (type(field)): variable to copy from
  !   to_field (type(field)): variable to copy to
  subroutine copy_fields(from_field, to_field)

    implicit none

    type(field), intent(in) :: from_field
    type(field), intent(out) :: to_field

    integer :: i, j, k

    ! Consistency checks
    if (.not.allocated(from_field%data)) then
       write (*,*) "Can not copy from a field without allocated data"
       stop
    end if
    if (.not.allocated(to_field%data)) then
       ! Target is not initialize, allocate memory
       allocate(to_field%data(lbound(from_field%data, 1):ubound(from_field%data, 1), &
            & lbound(from_field%data, 2):ubound(from_field%data, 2), &
            & lbound(from_field%data, 3):ubound(from_field%data, 3)))
    else if (any(shape(from_field%data) /= shape(to_field%data))) then
       write (*,*) "Wrong field data sizes in copy routine"
       print *, shape(from_field%data), shape(to_field%data)
       stop
    end if

    to_field%nx = from_field%nx
    to_field%ny = from_field%ny
    to_field%nz = from_field%nz
    to_field%nx_full = from_field%nx_full
    to_field%ny_full = from_field%ny_full
    to_field%nz_full = from_field%nz_full
    to_field%dx = from_field%dx
    to_field%dy = from_field%dy
    to_field%dz = from_field%dz

    !$omp parallel do private(i, j, k) collapse(2)
    do k = 0, from_field%nz + 1
       do j = 0, from_field%ny + 1
          do i = 0, from_field%nx + 1
             to_field%data(i, j, k) = from_field%data(i, j, k)
          end do
       end do
    end do
    !$omp end parallel do

  end subroutine copy_fields

  function average(field0) 
    use mpi

    implicit none

    real(dp) :: average
    type(field) :: field0

    real(dp) :: local_average
    integer :: rc

    local_average = sum(field0%data(1:field0%nx, 1:field0%ny, 1:field0%nz))
    call mpi_allreduce(local_average, average, 1, MPI_DOUBLE_PRECISION, MPI_SUM,  &
               &       MPI_COMM_WORLD, rc)
    average = average / (field0%nx_full * field0%ny_full * field0%nz_full)
    
  end function average

end module utilities
