import React from 'react';
import TodoItem from '../TodoItem/todoItem.js';

class TodoList extends React.Component {

    render () {
        const { todos } = this.props;
        return(
            <section className='todoListContainer'>
                {
                    todos.map((_todo, _index) => {
                        return(
                        <TodoItem updateTodoFn={this.updateTodo} key={_index} todo={_todo}></TodoItem>
                        )
                    })
                }
            </section>
        );
    }

    updateTodo = (todo) => {
        this.props.updateTodoFn(todo);
    }
}

export default TodoList